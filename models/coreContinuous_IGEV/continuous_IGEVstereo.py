import torch
import torch.nn as nn
import torch.nn.functional as F
from models.coreContinuous_IGEV.update import BasicMultiUpdateBlock
from models.coreContinuous_IGEV.extractor import MultiBasicEncoder, Feature
from models.coreContinuous_IGEV.geometry import Combined_Geo_Encoding_Volume
from models.coreContinuous_IGEV.submodule import *
import time
from models.coreContinuous_IGEV.liif import *

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class continuous_IGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.name_buff=[]
        context_dims = args.hidden_dims
        self.max_disp=args.max_disp
        self.multi_training=args.multi_training
        self.multi_input_training=args.multi_input_training
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        self.agg_type=args.agg_type
        self.feature = Feature()
        if 'type1' in args.agg_type:
            self.stem_2 = nn.Sequential(
                nn.PixelUnshuffle(2),
                BasicConv_IN(12, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(32), nn.ReLU()
                )
            self.stem_4 = nn.Sequential(
                nn.PixelUnshuffle(2),
                BasicConv_IN(128, 48, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(48), nn.ReLU()
                )   
            indim=48+32
            chanels=[48+args.hidden_dims[2],32]            
        elif 'type3' in args.agg_type:        
            self.stem_2 = HighRes_Aggregation(3,32)
            self.stem_4 = HighRes_Aggregation(32,48)
            indim=48+32
            chanels=[48+args.hidden_dims[2],32]            
        elif 'type4' in args.agg_type:        
            self.stem_2 = HighRes_Aggregation_LN(3,32)
            self.stem_4 = HighRes_Aggregation_LN(32,48)
            indim=48+32
            chanels=[48+args.hidden_dims[2],32]
        elif 'type5' in args.agg_type:        
            self.stem_2 = HighRes_Aggregation_LN_GeLU(3,32)
            self.stem_4 = HighRes_Aggregation_LN_GeLU(32,48)
            indim=48+32
            chanels=[48+args.hidden_dims[2],32]       
        elif 'type2' in args.agg_type:
            self.stem_1 = nn.Sequential(
                BasicConv_IN(3, 8, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(8, 8, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(8), nn.ReLU()
                )            
            self.stem_2 = nn.Sequential(
                nn.PixelUnshuffle(2),
                BasicConv_IN(32, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(32), nn.ReLU()
                )
            self.stem_4 = nn.Sequential(
                nn.PixelUnshuffle(2),
                BasicConv_IN(128, 48, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(48), nn.ReLU()
                )   
            indim=48+32+8
            chanels=[8,32,48+args.hidden_dims[2] ]
            print("type2 aggregation on     ^_^!")   
        else:
            assert False
        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        indim = indim+args.hidden_dims[2]
        affinity_settings = {}
        affinity_settings['win_w'] = args.lsp_width
        affinity_settings['win_h'] = args.lsp_height
        affinity_settings['dilation'] = args.lsp_dilation
        if self.multi_training or self.multi_input_training:
            self.liif_up = liif_out_multi_scale_Training(encoder_dim=indim, mlphidden_list=args.mlphidden_list, pos_dim=args.pos_dim, pos_enconding=args.pos_enconding, pos_enconding_new=args.pos_enconding_new,
                                                        local_ensemble=args.local_ensemble, decode_cell=args.decode_cell, unfold=args.unfold_Lac, affinity_settings=affinity_settings, quater_nearest=args.quater_nearest, require_grad=args.require_grad, number_input=3 if 'type2' in args.agg_type else 2, chanels=chanels)
        else:
            self.liif_up = liif_out(encoder_dim=args.Raw_Mask_dim+48, mlphidden_list=args.mlphidden_list, pos_dim=args.pos_dim, pos_enconding=args.pos_enconding,
                                   local_ensemble=args.local_ensemble, decode_cell=args.decode_cell, unfold=args.unfold, require_grad=args.require_grad)
        self.corr_stem = BasicConv(
            8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self):
        for name,m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                self.name_buff.append(name)

            if isinstance(m, nn.SyncBatchNorm):
                assert len(self.name_buff)>1
                if name in self.name_buff:
                    m.eval()
                    m.affine=False
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False                

    def upsample_disp(self, disp, hidden_layer, stem_4x, stem_2x, stem_1x, hr_coord=None, scale=1):

        with autocast(enabled=self.args.mixed_precision):
            x = torch.cat((stem_4x, hidden_layer), 1)    #[B,128+48,H/4,W/4]        
            
            h,w=disp.shape[-2:]
            if self.args.disparity_norm:
                disp=disp/w              
            elif self.args.disparity_norm2:
                disp=disp/w*1024 
            else:
                if self.multi_training or self.multi_input_training:
                    disp=disp*4.*(scale.view(-1,1,1,1))
                else:
                    disp=disp*4.*scale
            if self.multi_training or self.multi_input_training:
                if stem_1x==None:
                    up_mask=self.liif_up([x,stem_2x],hr_coord,scale)     # [B,9,H*W]  
                else:
                    up_mask=self.liif_up([stem_1x,stem_2x,x],hr_coord,scale)     # [B,                
                up_mask=F.softmax(up_mask, 1)      # [B,9,H*W]
                if self.args.quater_nearest==None:
                    up_disp = context_upsample_multiscale_train(disp, up_mask,hr_coord).unsqueeze(1)
                else:
                    up_disp = context_upsample_multiscale_train_quaterp(disp, up_mask,hr_coord).unsqueeze(1)

            else:
                up_mask=self.liif_up(x,tar_size=[int(h*4*scale),int(w*4*scale)])     # [B,9,H,W]
                up_mask=F.softmax(up_mask, 1)      # [B,9,H,W]
                up_disp = context_upsample(disp, up_mask,scale).unsqueeze(1)
            
            if self.args.disparity_norm:        #denorm
                if self.multi_training or self.multi_input_training:                
                    up_disp=up_disp*torch.round(w*4.*scale.view(-1,1,1))      
                else:
                    up_disp=up_disp*round(w*4.*scale)
                
                return up_disp      #[B,1,H*W]                 
            elif self.args.disparity_norm2:
                if self.multi_training or self.multi_input_training:                         
                    up_disp=up_disp/1024*torch.round(w*4.*scale.view(-1,1,1))             
                else:
                    up_disp=up_disp*round(w*4.*scale)
                return up_disp      #[B,H,W]   
            else:
                return up_disp      #[B,H,W]                     

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False,hr_coord=None,scale=1.0,output_raw=None):         
        """ Estimate disparity between pair of frames """
        
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision):
            features_left = self.feature(image1)    # [B,48,H/4,W/4] [B,64,H/8,W/8] [B,192,H/16,W/16] [B,160,H/32,W/32]
            features_right = self.feature(image2)
            if 'type2' in self.agg_type:
                stem_1x=self.stem_1(image1)
                stem_2x = self.stem_2(stem_1x)           
                stem_1y=self.stem_1(image2)
                stem_2y = self.stem_2(stem_1y)
            else:
                stem_1x=None
                stem_2x = self.stem_2(image1) 
                stem_2y = self.stem_2(image2)
            stem_4x = self.stem_4(stem_2x)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)    # [B,48+48,H/4,W/4]
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)  # [B,48+48,H/4,W/4]
            match_left = self.desc(self.conv(features_left[0]))             # [B,96,H/4,W/4]
            match_right = self.desc(self.conv(features_right[0]))           # [B,96,H/4,W/4]
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8) 
            gwc_volume = self.corr_stem(gwc_volume)             #[B,8,48,H/4,W/4] 
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])    #[B,8,48,H/4,W/4]  dot by the attention
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)      
            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)       
            init_disp = disparity_regression(prob, self.args.max_disp//4)        
            del prob, gwc_volume
            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)        
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        geo_block = Combined_Geo_Encoding_Volume            # mesh the features.
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        
        
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []
        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)         
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res ConvGRU and mid-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers==3, iter08=True, iter04=False, update=False)
                                                            #          hidden    context  corr  Disparity iter32 iter16
                net_list, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)

            disp = disp + delta_disp       
            if test_mode and itr < iters-1:
                continue

            # upsample predictions            #[B,32,H/4,W/4] #[B,48,H/4,W/4]
            disp_up = self.upsample_disp(disp,net_list[0],stem_4x,stem_2x,stem_1x,hr_coord=hr_coord,scale=scale) 
            disp_preds.append(disp_up)
        
        if test_mode:           # if in test model directly upload disp_up
            return disp_up
        return  init_disp.squeeze(1),disp_preds       
