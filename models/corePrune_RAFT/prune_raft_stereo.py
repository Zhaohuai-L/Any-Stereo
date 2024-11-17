import torch
import torch.nn as nn
import torch.nn.functional as F
from models.corePrune_RAFT.update import BasicMultiUpdateBlock
from models.corePrune_RAFT.extractor import MultiBasicEncoder,BasicEncoder
from models.corePrune_RAFT.geometry import CorrBlock1D
from models.corePrune_RAFT.submodule import *
import time

from models.corePrune_RAFT.liif import *

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

    def forward(self, x, features):     #These are different from the GwcNet, which may got from his teacher.
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

class continuous_RaftStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        #AnyStereo    
        self.multi_training=args.multi_training
        self.multi_input_training=args.multi_input_training
        self.agg_type=args.agg_type
        
        #RAFT-stereo
        context_dims = args.hidden_dims
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

        if 'IGEV' in args.agg_type:
            self.stem_2 = nn.Sequential(
                BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(32), nn.ReLU()
                )
            self.stem_4 = nn.Sequential(
                BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(48), nn.ReLU()
                )
            indim=48+32
        elif 'type1' in args.agg_type:      
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
            print("type1 aggregation on     ^_^!")   
        elif 'type3' in args.agg_type:        
            self.stem_2 = HighRes_Aggregation(3,32)
            self.stem_4 = HighRes_Aggregation(32,48)
            indim=48+32
            chanels=[48+args.hidden_dims[2],32]            
            print("type3 aggregation on     ^_^!")   
        elif 'type4' in args.agg_type:        
            self.stem_2 = HighRes_Aggregation_LN(3,32)
            self.stem_4 = HighRes_Aggregation_LN(32,48)
            indim=48+32
            chanels=[48+args.hidden_dims[2],32]
            print("type4 aggregation on     ^_^!")       
        elif 'type5' in args.agg_type:        
            self.stem_2 = HighRes_Aggregation_LN_GeLU(3,32)
            self.stem_4 = HighRes_Aggregation_LN_GeLU(32,48)
            indim=48+32
            chanels=[48+args.hidden_dims[2],32]
            print("type5 aggregation on     ^_^!")                        
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
            indim=0
            chanels=[args.hidden_dims[2]]

        indim=indim+args.hidden_dims[2]    
        affinity_settings = {}
        affinity_settings['win_w'] = args.lsp_width
        affinity_settings['win_h'] = args.lsp_width
        affinity_settings['dilation'] = args.lsp_dilation
        if self.multi_training or self.multi_input_training:
            self.liif_up=liif_out_multi_scale_Training(encoder_dim=indim,mlphidden_list=args.mlphidden_list,pos_dim=args.pos_dim,pos_enconding=args.pos_enconding,pos_enconding_new=args.pos_enconding_new,
                             local_ensemble=args.local_ensemble,decode_cell=args.decode_cell, unfold=args.unfold_Lac,affinity_settings=affinity_settings,quater_nearest=args.quater_nearest,require_grad=args.require_grad,number_input= len(chanels),chanels=chanels)
        else:
            self.liif_up=liif_out(encoder_dim=args.Raw_Mask_dim+48,mlphidden_list=args.mlphidden_list,pos_dim=args.pos_dim,pos_enconding=args.pos_enconding,
                             local_ensemble=args.local_ensemble,decode_cell=args.decode_cell, unfold=args.unfold,require_grad=args.require_grad)            
            
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, hidden_layer, stem_4x, stem_2x, stem_1x, hr_coord=None, scale=1):

        with autocast(enabled=self.args.mixed_precision):
            if stem_4x!=None:
                x = torch.cat((stem_4x, hidden_layer), 1) 
            else:
                x=hidden_layer
            h,w=disp.shape[-2:]
            if self.args.disparity_norm:
                disp=disp/w
            else:
                if self.multi_training or self.multi_input_training:
                    disp=disp*4.*(scale.view(-1,1,1,1))
                else:
                    disp=disp*4.*scale
            if self.multi_training or self.multi_input_training:
                if stem_1x!=None :
                    up_mask=self.liif_up([stem_1x,stem_2x,x],hr_coord,scale)         
                elif stem_2x!=None :
                    up_mask=self.liif_up([x,stem_2x],hr_coord,scale) 
                else:
                    up_mask=self.liif_up([x],hr_coord,scale) 
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
            else:
                return up_disp      #[B,H,W]                     



    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False,hr_coord=None,scale=1.0,output_raw=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision):
            match_left, match_right = self.fnet([image1, image2])
            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)        
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]
            if 'None' not in self.agg_type:
                if 'type2' in self.agg_type:
                    stem_1x=self.stem_1(image1)
                    stem_2x = self.stem_2(stem_1x)
                else:
                    stem_1x=None
                    stem_2x = self.stem_2(image1) 
                stem_4x = self.stem_4(stem_2x)           
            else:
                stem_4x=stem_2x=stem_1x=None
        corr_block = CorrBlock1D            # mesh the features.
        corr_fn = corr_block(match_left.float(), match_right.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        
        
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        # disp = init_disp
        disp = match_left.new_zeros((b,1,h,w))
        disp_preds = []
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = corr_fn(disp, coords) 
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers==3, iter08=True, iter04=False, update=False)
                net_list, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            disp_up = self.upsample_disp(disp,net_list[0],stem_4x,stem_2x,stem_1x,hr_coord=hr_coord,scale=scale)
            disp_preds.append(disp_up)
        if test_mode:
            if output_raw:
                return disp,disp_up                
            else:                
                return disp_up
        return  disp_preds 
