from xml.etree.ElementInclude import include
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.coreContinuous_A2A4IGEV.submodule import *
import timm
import math


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)



def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))
    


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float() 
        coord_seqs.append(seq)      #[H,W]
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)      #[H,W,2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1])       #[H*W,2]
    return ret


def liif_feat(feat, target_size, local=False, cell=False):
    bs, l_h, l_w = feat.shape[0], feat.shape[-2], feat.shape[-1]
    h_h, h_w = target_size
    coords = make_coord((h_h, h_w)).cuda()
    coords = coords.expand(bs, *coords.shape)  # [B,H*W,2]
    if cell:
        cells = torch.ones_like(coords)  # [B,H*W,2]
        cells[:, :, 0] = 2/h_h
        cells[:, :, 1] = 2/h_w
    # [B,2,l_h,l_w]
    feat_coords = make_coord((l_h, l_w), flatten=False).cuda().permute(
        2, 0, 1) .unsqueeze(0).expand(bs, 2, l_h, l_w)

    if local:
        vx_list = [-1, 1]
        vy_list = [-1, 1]
        eps_shift = 1e-6
        rel_coord_list = []
        q_feat_list = []
        area_list = []
        rel_cell_list = []
    else:
        vx_list, vy_list, eps_shift = [0], [0], 0  

    # field radius (global: [-1, 1])
    rx = 2 / feat.shape[-2] / 2     # 1/W
    ry = 2 / feat.shape[-1] / 2     # 1/H

    for vx in vx_list:
        for vy in vy_list:
            coords_ = coords.clone()        # [B,H*W,2]
            coords_[:, :, 0] += vx * rx + eps_shift
            coords_[:, :, 1] += vy * ry + eps_shift
            coords_.clamp_(-1+1e-6, 1-1e-6)
            coords_ = coords_.flip(-1).unsqueeze(1)     # [B,1,H*W,2]

            q_feat = F.grid_sample(feat, coords_, mode='nearest', align_corners=False)[
                :, :, 0, :].permute(0, 2, 1)  # [B,H*W,C]
            q_coord = F.grid_sample(feat_coords, coords_, mode='nearest', align_corners=False)[
                :, :, 0, :].permute(0, 2, 1)  # [B,H*W,C(2)]

            rel_coord = coords - q_coord
            rel_coord[:, :, 0] *= l_h
            rel_coord[:, :, 1] *= l_w
            if cell:
                rel_cell = cells.clone()
                rel_cell[:, :, 0] *= l_h
                rel_cell[:, :, 1] *= l_w
            if local:
                rel_coord_list.append(rel_coord)
                q_feat_list.append(q_feat)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                area_list.append(area+1e-9)
                if cell:
                    rel_cell_list.append(rel_cell)

    if not local:
        if cell:
            return rel_coord, q_feat, rel_cell        # [B,H*W,2]  [B,H*W,C]
        else:
            return rel_coord, q_feat, None        # [B,H*W,2]  [B,H*W,C]
    else:
        # [4*[B,H*W,2]]  [4*[B,H*W,C]]  [4*[B,H*W]]
        return rel_coord_list, q_feat_list, area_list, rel_cell_list


def liif_feat_multiscale_train(feat, coords, scale, local=False, cell=False):
    # l_h means low resolution height
    bs, l_h, l_w = feat.shape[0], feat.shape[-2], feat.shape[-1]

    if cell:
        cells = torch.ones_like(coords)  # [B,H*W,2]
        cells[:, :, 0] = 2/(scale)
        cells[:, :, 1] = 2/(scale)
    # [B,2,l_h,l_w]
    feat_coords = make_coord((l_h, l_w), flatten=False).cuda().permute(
        2, 0, 1) .unsqueeze(0).expand(bs, 2, l_h, l_w)
    coords_ = coords.clone()        # [B,H*W,2]
    # coords_[:,:,0] += vx * rx + eps_shift
    # coords_[:,:,1] += vy * ry + eps_shift
    coords_.clamp_(-1+1e-6, 1-1e-6)  # clamp
    coords_ = coords_.flip(-1).unsqueeze(1)     # [B,1,H*W,2]

    q_feat = F.grid_sample(feat, coords_, mode='nearest', align_corners=False)[
        :, :, 0, :].permute(0, 2, 1)  # [B,H*W,C]
    q_coord = F.grid_sample(feat_coords, coords_, mode='nearest', align_corners=False)[
        :, :, 0, :].permute(0, 2, 1)  # [B,H*W,C(2)]
    rel_coord = coords - q_coord
    rel_coord[:, :, 0] *= l_h
    rel_coord[:, :, 1] *= l_w

    if not local:
        if cell:
            return rel_coord, q_feat, cells        # [B,H*W,2]  [B,H*W,C]
        else:
            return rel_coord, q_feat, None        # [B,H*W,2]  [B,H*W,C]
    else:
        assert False

def liif_feat_multiscale_train_quater(feat, coords,scale, local=False,cell=False):
    """ Computing the upfeat and relative coord.
        feat: tensor [B,C,H,W] the latent code, i.e., the feature from Input low res image.
        ret: 
    """
    bs, l_h, l_w = feat.shape[0], feat.shape[-2], feat.shape[-1]        #l_h means low resolution height

    if cell:
        cells= torch.ones_like(coords)   #[B,H*W,2]
        cells[:,:,0]=2/(scale)
        cells[:,:,1]=2/(scale)
        
        
    vx_lst = [-1, 1] 
    vy_lst = [-1, 1]
    eps_shift = 1e-6
    rx = 2 / feat.shape[-2] / 2     # 1/W 
    ry = 2 / feat.shape[-1] / 2     # 1/H 
    q_feats=[]
    q_coords=[]
    #[B,2,l_h,l_w]            
    feat_coords = make_coord((l_h,l_w), flatten=False).cuda().permute(2, 0, 1) .unsqueeze(0).expand(bs, 2, l_h,l_w)

    for vx in vx_lst:
        for vy in vy_lst:
            coords_ = coords.clone()        # [B,H*W,2]
            coords_[:,:,0] += vx * rx + eps_shift
            coords_[:,:,1] += vy * ry + eps_shift
            coords_.clamp_(-1+1e-6, 1-1e-6) 
            coords_=coords_.flip(-1).unsqueeze(1)     # [B,1,H*W,2]    
            
            q_feat = F.grid_sample(feat, coords_,mode='nearest',align_corners=False)[:,:,0,:].permute(0,2,1)    #[B,H*W,C]
            q_coord = F.grid_sample(feat_coords, coords_,mode='nearest',align_corners=False)[:,:,0,:].permute(0,2,1) #[B,H*W,C(2)] 
            q_feats.append(q_feat)
            q_coords.append(q_coord)
    q_center_coord=(q_coords[0]+q_coords[3])/2 
    q_quater_feat=torch.cat(q_feats,dim=-1) # [B,4,H*W]
    del q_coords,q_feats
    rel_coord = coords - q_center_coord
    rel_coord[:,:,0] *= l_h
    rel_coord[:,:,1] *= l_w

    if cell:
        return rel_coord, q_quater_feat,cells        # [B,H*W,2]  [B,H*W,C]  
    else:
        return rel_coord, q_quater_feat,None        # [B,H*W,2]  [B,H*W,C]  

class PositionEncoder(nn.Module):
    def __init__(
        self,
        posenc_type=None,
        complex_transform=False,
        posenc_scale=6,
        gauss_scale=1,
        in_dims=2, 
        enc_dims=256,
        hidden_dims=32,
        head=1,
        gamma=1
    ):
        super().__init__()

        self.posenc_type = posenc_type
        self.complex_transform = complex_transform
        self.posenc_scale = posenc_scale
        self.gauss_scale = gauss_scale

        self.in_dims = in_dims
        self.enc_dims = enc_dims
        self.hidden_dims = hidden_dims
        self.head = head
        self.gamma = gamma

        self.define_parameter()

    def define_parameter(self):
        if self.posenc_type == 'sinusoid' or self.posenc_type == 'ipe':
            self.b_vals = 2.**torch.linspace(
                0, self.posenc_scale, self.enc_dims // 4
            ) - 1  # -1 -> (2 * pi)
            self.b_vals = torch.stack([self.b_vals, torch.zeros_like(self.b_vals)], dim=-1)
            self.b_vals = torch.cat([self.b_vals, torch.roll(self.b_vals, 1, -1)], dim=0)
            self.a_vals = torch.ones(self.b_vals.shape[0])
            self.proj = nn.Linear(self.enc_dims, self.head)
        elif self.posenc_type == 'learn':
            self.Wr = nn.Linear(self.in_dims, self.hidden_dims // 2, bias=False)
            self.mlp = nn.Sequential(
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.Linear(self.hidden_dims, self.hidden_dims),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.Linear(self.hidden_dims, self.enc_dims)
            )
            self.proj = nn.Sequential(nn.GELU(), nn.Linear(self.enc_dims, self.head))
            self.init_weight()

        elif self.posenc_type == 'dpb':
            self.mlp = nn.Sequential(
                nn.Linear(2, self.hidden_dims),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.hidden_dims, self.hidden_dims),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.hidden_dims, self.enc_dims)
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.enc_dims, self.head)
            )

    def init_weight(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, positions, cells=None):

        if self.posenc_type is None:
            return positions

        if self.posenc_type == 'sinusoid' or self.posenc_type == 'ipe':
            self.b_vals = self.b_vals.cuda()
            self.a_vals = self.a_vals.cuda()

            # b, q, 1, c (x -> c/2, y -> c/2)
            sin_part = self.a_vals * torch.sin(
                torch.matmul(positions, self.b_vals.transpose(-2, -1))
            )
            cos_part = self.a_vals * torch.cos(
                torch.matmul(positions, self.b_vals.transpose(-2, -1))
            )

            if self.posenc_type == 'ipe':
                # b, q, 2
                cell = cells.clone()
                cell_part = torch.sinc(
                    torch.matmul((1 / np.pi * cell), self.b_vals.transpose(-2, -1))
                )

                sin_part = sin_part * cell_part
                cos_part = cos_part * cell_part

            if self.complex_transform:
                pos_enocoding = torch.view_as_complex(torch.stack([cos_part, sin_part], dim=-1))
            else:
                pos_enocoding = torch.cat([sin_part, cos_part], dim=-1)
                pos_bias = self.proj(pos_enocoding)

        elif self.posenc_type == 'learn':
            projected_pos = self.Wr(positions)

            sin_part = torch.sin(projected_pos)
            cos_part = torch.cos(projected_pos)

            if self.complex_transform:
                pos_enocoding = 1 / np.sqrt(self.hidden_dims) * torch.view_as_complex(
                    torch.stack([cos_part, sin_part], dim=-1)
                )
            else:
                pos_enocoding = 1 / np.sqrt(self.hidden_dims
                                           ) * torch.cat([sin_part, cos_part], dim=-1)
                pos_enocoding = self.mlp(pos_enocoding)

        elif self.posenc_type == 'dpb':
            pos_enocoding = self.mlp(positions)

        pos_bias = None if self.complex_transform else self.proj(pos_enocoding)

        return pos_enocoding, pos_bias



class SpatialEncoding(nn.Module):       
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 sigma = 6,
                 cat_input=True,
                 require_grad=False,):

        super().__init__()
        assert out_dim % (2*in_dim) == 0, "dimension must be dividable"

        n = out_dim // 2 // in_dim
        m = 2**np.linspace(0, sigma, n)
        m = np.stack([m] + [np.zeros_like(m)]*(in_dim-1), axis=-1)
        m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], axis=0)
        self.emb = torch.FloatTensor(m)    
        if require_grad:
            self.emb = nn.Parameter(self.emb, requires_grad=True)    
        self.in_dim = in_dim    
        self.out_dim = out_dim  
        self.sigma = sigma  
        self.cat_input = cat_input
        self.require_grad = require_grad

    def forward(self, x):
        if not self.require_grad:
            self.emb = self.emb.to(x.device)
        y = torch.matmul(x, self.emb.T)     #
        if self.cat_input:
            return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)


class liif_out(nn.Module):
    def __init__(self, pos_dim=24,encoder_dim=256,mlphidden_list=[128,64,64],pos_enconding=False,local_ensemble=False,decode_cell=False, unfold=False,require_grad=True):
        super(liif_out, self).__init__()
        self.local_ensemble = local_ensemble
        self.pos_dim=pos_dim
        self.encoder_dim=encoder_dim
        self.unfold=unfold
        self.decode_cell=decode_cell
        self.pos_enconding=pos_enconding
        self.pos_encoding = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
        
        if self.pos_enconding:
            self.pos_dim=pos_dim+2 
        else:
            self.pos_dim=2 
        imnet_in_dim = self.encoder_dim    
        if self.unfold:  
            imnet_in_dim = imnet_in_dim*9
        imnet_in_dim=imnet_in_dim + self.pos_dim   
        if self.decode_cell:
            imnet_in_dim=imnet_in_dim+2
        
        self.imnet = MLP((imnet_in_dim),3*3,hidden_list=mlphidden_list)

    def forward(self, feat, tar_size):      #[B,80,H/4,W/4]   [H,W]
        h_h ,h_w= tar_size
        b,c,h,w=feat.shape
        if not self.local_ensemble:      
            if self.unfold:  
                feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])    #[B,9C,H/4,W/4]                          
            rel_coord, q_feat,rel_cell=liif_feat(feat,tar_size,self.local_ensemble,self.decode_cell)        # [B,H*W,2]  [B,H*W,C]        
            if self.pos_enconding:
                rel_coord = self.pos_encoding(rel_coord) #Position Encoding 
            feat=torch.cat([q_feat,rel_coord],dim=-1)    #[B,H*W,C]
            del rel_coord,q_feat
            if self.decode_cell:
                feat=torch.cat([feat,rel_cell],dim=-1)    #[B,H*W,C]
            feat=feat.view(b*h_h*h_w,-1)    
            feat=self.imnet(feat).view(b,h_h*h_w,3*3)   #[B,H*W,C]
            feat=feat.permute(0,2,1) #[B,C,H*W] 
            feat=feat.view(b,-1,h_h,h_w)        #[B,9,H,W]
            return feat   
        else:
            if self.unfold:  
                feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])    #[B,9C,H/4,W/4]          
            rel_coord_list, q_feat_list, area_list,rel_cell_list = liif_feat(feat,tar_size, self.local_ensemble,self.decode_cell)    # [4*[B,H*W,2]]  [4*[B,H*W,C]]  [4*[B,H*W]]  
            preds = []  
            for q_feat,rel_coord,rel_cell in zip (q_feat_list,rel_coord_list,rel_cell_list) :
                if self.pos_enconding:
                    rel_coord = self.pos_encoding(rel_coord) #Position Encoding 
                q_feat=torch.cat([q_feat,rel_coord],dim=-1)    #[B,H*W,C]    
                if self.decode_cell:
                    q_feat=torch.cat([q_feat,rel_cell],dim=-1)    #[B,H*W,C]                          
                q_feat=q_feat.view(b*h_h*h_w,-1)        
                q_feat=self.imnet(q_feat).view(b,h_h*h_w,3*3)   #[B,H*W,C]                                                          
                preds.append(q_feat)
                
            tot_area = torch.stack(area_list).sum(dim=0)# [4,B,H*W]               
            t = area_list[0]; area_list[0] = area_list[3]; area_list[3] = t
            t = area_list[1]; area_list[1] = area_list[2]; area_list[2] = t                
            ret = 0
            for pred, area in zip(preds, area_list):
                ret = ret + pred * (area / tot_area).unsqueeze(-1)          
            ret=ret.permute(0,2,1) #[B,C,H*W]        
            ret=ret.view(b,-1,h_h,h_w)        #[B,9,H,W]                       
            return ret
  
  
class AffinityFeature(nn.Module):
    def __init__(self, win_h, win_w, dilation, cut):
        super(AffinityFeature, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.dilation = dilation
        self.cut = 0
        self._padding = win_w//2

    def padding(self, x, win_h, win_w, dilation):
        pad_t = (win_w // 2 * dilation, win_w // 2 * dilation,
                 win_h // 2 * dilation, win_h // 2 * dilation)
        out = F.pad(x, pad_t, mode='constant')
        return out

    def forward(self, feature):
        B, C, H, W = feature.size()
        feature = F.normalize(feature, dim=1, p=2)

        unfold_feature = nn.Unfold(
            kernel_size=(self.win_h, self.win_w), dilation=self.dilation, padding=self._padding)(feature)
        all_neighbor = unfold_feature.reshape(B, C, -1, H, W).transpose(1, 2)
        num = (self.win_h * self.win_w) // 2
        neighbor = torch.cat(
            (all_neighbor[:, :num], all_neighbor[:, num+1:]), dim=1)
        feature = feature.unsqueeze(1)  # [B,1,C,H,W]
        affinity = torch.sum(neighbor * feature, dim=2)  # [B,8,H,W]
        affinity[affinity < self.cut] = self.cut

        return affinity


class StructureFeature(nn.Module):
    def __init__(self, affinity_settings, unfold,input_chanels):
        super(StructureFeature, self).__init__()

        self.win_w = affinity_settings['win_w']
        self.win_h = affinity_settings['win_h']
        self.dilation = affinity_settings['dilation']
        self.unfold=unfold
        in_c = self.win_w * self.win_h - 1
        self.Affi1=AffinityFeature(self.win_h, self.win_w, self.dilation[0], 0)    #[B,8,H,W]       
        if "Dila_ISU" in  self.unfold: 
            self.Affi2=AffinityFeature(self.win_h, self.win_w, self.dilation[1], 0)    #[B,8,H,W]       
            self.Affi3=AffinityFeature(self.win_h, self.win_w, self.dilation[2], 0)    #[B,8,H,W]   
            self.Affi4=AffinityFeature(self.win_h, self.win_w, self.dilation[3], 0)    #[B,8,H,W]   
            self.sfc_conv1 = nn.Sequential(convbn(in_c, in_c, 1, 1, 0, 1),         
                                           nn.ReLU(inplace=True))
            self.sfc_conv2 = nn.Sequential(convbn(in_c, in_c, 1, 1, 0, 1), 
                                           nn.ReLU(inplace=True))
            self.sfc_conv3 = nn.Sequential(convbn(in_c, in_c, 1, 1, 0, 1),
                                           nn.ReLU(inplace=True))
            self.sfc_conv4 = nn.Sequential(convbn(in_c, in_c, 1, 1, 0, 1),
                                           nn.ReLU(inplace=True))
        elif "Dila_3ISU" in  self.unfold: 
            self.sfc_embeding =convbn(input_chanels, input_chanels//4, 1, 1, 0, 1)                                 
            self.Affi2=AffinityFeature(self.win_h, self.win_w, self.dilation[1], 0)    #[B,8,H,W]       
            self.Affi3=AffinityFeature(self.win_h, self.win_w, self.dilation[2], 0)    #[B,8,H,W]    
        elif "Dila_2ISU" in  self.unfold: 
            self.sfc_embeding =convbn(input_chanels, input_chanels//4, 1, 1, 0, 1)         
                                
            self.Affi2=AffinityFeature(self.win_h, self.win_w, self.dilation[1], 0)    #[B,8,H,W]         
        elif "with_1_43ISU" in  self.unfold:                               
            self.Affi2=AffinityFeature(self.win_h, self.win_w, self.dilation[1], 0)    #[B,8,H,W]       
            self.Affi3=AffinityFeature(self.win_h, self.win_w, self.dilation[2], 0)    #[B,8,H,W]          
            self.sfc_conv1 = nn.Sequential(convbn(in_c, in_c//2, 1, 1, 0, 1),         
                                nn.ReLU(inplace=True))      
            self.sfc_conv2 = nn.Sequential(convbn(in_c, in_c//2, 1, 1, 0, 1),         
                                nn.ReLU(inplace=True))     
            self.sfc_conv3 = nn.Sequential(convbn(in_c, in_c//2, 1, 1, 0, 1),         
                                nn.ReLU(inplace=True))            
        elif "with_1_43v2ISU" in  self.unfold or "with_3v2ISU" in  self.unfold:                               
            self.Affi2=AffinityFeature(self.win_h, self.win_w, self.dilation[1], 0)    #[B,8,H,W]       
            self.Affi3=AffinityFeature(self.win_h, self.win_w, self.dilation[2], 0)    #[B,8,H,W]                                      
        elif "with_embed_ISU" in self.unfold:  
            self.sfc_embeding =convbn(input_chanels+8, input_chanels+8, 1, 1, 0, 1)       
    def forward(self, x):       #[B,C,H,W]
        if "with_ISU" in self.unfold: 
            affinity1 = self.Affi1(x)    #[B,8,H,W]       
            x=torch.cat([x,affinity1],dim=1)          #[B,C+(winH*winW-1),H,W]
        elif "with_v2ISU" in self.unfold: 
            feat=x.detach()
            affinity1 = self.Affi1(feat)    #[B,8,H,W]       
            x=torch.cat([x,affinity1], dim=1)          #[B,C+(winH*winW-1),H,W]            
        elif "with_1_4ISU" in self.unfold:  
            # feat=self.sfc_embeding(x) 
            affinity1 = self.Affi1(x)    #[B,8,H,W]       
            x=torch.cat([x,affinity1],dim=1)          #[B,C+(winH*winW-1),H,W]  
        elif "with_1_43ISU" in self.unfold:  
            feat=x.detach()
            affinity1 = self.Affi1(feat)      #[B,8,H,W]       
            affinity2 = self.Affi2(feat)      #[B,8,H,W]  
            affinity3 = self.Affi3(feat)      #[B,8,H,W]       
            affi_feature1 = self.sfc_conv1(affinity1)                   
            affi_feature2 = self.sfc_conv2(affinity2)
            affi_feature3 = self.sfc_conv3(affinity3)
            x=torch.cat([x,affi_feature1,affi_feature2,affi_feature3],dim=1)          #[B,C+(winH*winW-1),H,W]
        elif "with_1_43v2ISU" in self.unfold:  
            feat=x.detach()
            affinity1 = self.Affi1(feat)      #[B,8,H,W]
            affinity2 = self.Affi2(feat)      #[B,8,H,W]        
            affinity3 = self.Affi3(feat)      #[B,8,H,W]        
            x=torch.cat([x,affinity1,affinity2,affinity3],dim=1)          #[B,C+(winH*winW-1),H,W] 
        elif "with_3v2ISU" in self.unfold:
            feat=x.detach()
            affinity1 = self.Affi1(feat)      #[B,8,H,W]         
            affinity2 = self.Affi2(feat)      #[B,8,H,W]        
            affinity3 = self.Affi3(feat)      #[B,8,H,W]        
            x=torch.cat([x,affinity1,affinity2,affinity3],dim=1)          #[B,C+(winH*winW-1),H,W]                                
        elif "with_embed_ISU" in self.unfold:  
            affinity1 = self.Affi1(x.detach())    #[B,8,H,W]
            feat_embed=torch.cat([x,affinity1],dim=1)          #[B,C+(winH*winW-1),H,W]    
            x=self.sfc_embeding(feat_embed) 
           
        elif "only_ISU" in self.unfold: 
            x=self.Affi1(x)    #[B,8,H,W] 
        elif 'with_Dila_ISU' in self.unfold:
            affinity1 = self.Affi1(x)      #[B,8,H,W]       
            affinity2 = self.Affi2(x)      #[B,8,H,W]  
            affinity3 = self.Affi3(x)      #[B,8,H,W]  
            affinity4 = self.Affi4(x)      #[B,8,H,W]  
            affi_feature1 = self.sfc_conv1(affinity1)               #[B,lsp_channel,H,W]
            affi_feature2 = self.sfc_conv2(affinity2)
            affi_feature3 = self.sfc_conv3(affinity3)
            affi_feature4 = self.sfc_conv4(affinity4)
            x = torch.cat((x,affi_feature1, affi_feature2, affi_feature3, affi_feature4), dim=1)
        elif "only_Dila_ISU" in self.unfold: 
            affinity1 = self.Affi1(x)      #[B,8,H,W]       
            affinity2 = self.Affi2(x)      #[B,8,H,W]  
            affinity3 = self.Affi3(x)      #[B,8,H,W]  
            affinity4 = self.Affi4(x)      #[B,8,H,W]  

            affi_feature1 = self.sfc_conv1(affinity1)               #[B,lsp_channel,H,W]
            affi_feature2 = self.sfc_conv2(affinity2)
            affi_feature3 = self.sfc_conv3(affinity3)
            affi_feature4 = self.sfc_conv4(affinity4)

            x = torch.cat((affi_feature1, affi_feature2, affi_feature3, affi_feature4), dim=1)
        elif 'with_Dila_2ISU' in self.unfold:
            feat=self.sfc_embeding(x) 
            affinity1 = self.Affi1(feat)      #[B,8,H,W]       
            affinity2 = self.Affi2(feat)      #[B,8,H,W]  
            x = torch.cat((x,affinity1, affinity2), dim=1)
        elif "only_Dila_2ISU" in self.unfold: 
            feat=self.sfc_embeding(x)             
            affinity1 = self.Affi1(feat)      #[B,8,H,W]       
            affinity2 = self.Affi2(feat)      #[B,8,H,W]  
            x = torch.cat((affinity1, affinity2), dim=1)               
        elif 'with_Dila_3ISU' in self.unfold:
            feat=self.sfc_embeding(x)                 
            affinity1 = self.Affi1(feat)      #[B,8,H,W]       
            affinity2 = self.Affi2(feat)      #[B,8,H,W]  
            affinity3 = self.Affi3(feat)      #[B,8,H,W]  

            x = torch.cat((x,affinity1, affinity2, affinity3), dim=1)
        elif "only_Dila_3ISU" in self.unfold: 
            feat=self.sfc_embeding(x)                 
            affinity1 = self.Affi1(feat)      #[B,8,H,W]       
            affinity2 = self.Affi2(feat)      #[B,8,H,W]  
            affinity3 = self.Affi3(feat)      #[B,8,H,W]  


            x = torch.cat((affinity1, affinity2, affinity3), dim=1)      
        return x


class liif_out_multi_scale_Training(nn.Module):
    def __init__(self, pos_dim=24,encoder_dim=256,mlphidden_list=[128,64,64],pos_enconding=False,pos_enconding_new=False,local_ensemble=False,decode_cell=False, unfold=False,affinity_settings=None,quater_nearest=None,require_grad=True,number_input=3,chanels=0):
        super(liif_out_multi_scale_Training, self).__init__()
        self.local_ensemble = local_ensemble
        self.pos_dim=pos_dim
        self.encoder_dim=encoder_dim
        self.unfold=unfold
        self.decode_cell=decode_cell
        self.pos_enconding=pos_enconding
        self.pos_enconding_new=pos_enconding_new
        self.quater_nearest=quater_nearest
        self.pos_dim=2 
        if pos_dim!=0:
            if self.pos_enconding:
                self.pos_encoding = SpatialEncoding(2, pos_dim, require_grad=require_grad) 
                self.pos_dim=pos_dim+2    
            elif self.pos_enconding_new:
                self.pos_encoding = PositionEncoder(posenc_type='sinusoid',posenc_scale=10,hidden_dims=self.pos_dim,enc_dims=self.pos_dim,head=8)
                self.pos_dim=8  
            else:
                self.pos_encoding = SpatialEncoding(2, pos_dim, require_grad=require_grad) 
        self.outputdim=3*3
        
        imnet_in_dim = self.encoder_dim    
        in_c = affinity_settings['win_h'] * affinity_settings['win_w'] - 1
        if self.unfold!=None:
            if ("with_1_4ISU" in self.unfold) or ("with_1_43ISU" in self.unfold)or ("with_1_43v2ISU" in self.unfold):
                self.to_sf_l2 = StructureFeature(affinity_settings, self.unfold,input_chanels=None) 
            else:
                self.to_sf_l2 = nn.ModuleList(StructureFeature(affinity_settings, self.unfold,input_chanels=i) for i in chanels
                                            )   
            if 'only_unfold' in self.unfold:  
                imnet_in_dim = imnet_in_dim*9
            elif "with_1_4ISU" in self.unfold:
                imnet_in_dim+=in_c
            elif "with_1_43ISU" in self.unfold:
                imnet_in_dim+=(in_c//2)*3          
            elif "with_1_43v2ISU" in self.unfold:
                imnet_in_dim+=(in_c)*3                   
            elif "with_3v2ISU" in self.unfold:
                imnet_in_dim+=(in_c*3*number_input)                                    
            elif "with_ISU" in self.unfold or "with_v2ISU" in self.unfold: 
                imnet_in_dim+=(in_c*number_input)
            elif "with_embed_ISU" in self.unfold: 
                imnet_in_dim+=(in_c*number_input)            
            elif "only_ISU" in self.unfold: 
                imnet_in_dim=in_c*number_input
            elif 'with_Dila_ISU' in self.unfold:
                imnet_in_dim+=(in_c*4*number_input) 
            elif "only_Dila_ISU" in self.unfold: 
                imnet_in_dim=(in_c*4*number_input) 
            elif 'with_Dila_3ISU' in self.unfold:
                imnet_in_dim+=(in_c*3*number_input) 
            elif "only_Dila_3ISU" in self.unfold: 
                imnet_in_dim=(in_c*3*number_input)             
            elif 'with_Dila_2ISU' in self.unfold:
                imnet_in_dim+=(in_c*2*number_input) 
            elif "only_Dila_2ISU" in self.unfold: 
                imnet_in_dim=(in_c*2*number_input)                        
            else:
                assert False 
        if quater_nearest!=None:
            self.outputdim=4
            if 'both' in self.quater_nearest:
                imnet_in_dim=imnet_in_dim*4
        imnet_in_dim=imnet_in_dim + self.pos_dim*number_input   
        if self.decode_cell:
            imnet_in_dim=imnet_in_dim+2*number_input
            
        self.imnet = MLP((imnet_in_dim),self.outputdim,hidden_list=mlphidden_list)   
        
    def forward(self, feats, coord,scale):      #[B,80,H/4,W/4]   [H,W]
        # h_h ,h_w= tar_size
        latent = []
        i=0
        for feat in feats:
            b,c,h,w=feat.shape
            bs,q=coord.shape[:2]  
            if self.unfold != None:      
                if 'only_unfold' in self.unfold:  
                    feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])    #[B,9C,H/4,W/4]                          
                elif (('with_1_4ISU' in self.unfold) or ('with_1_43ISU' in self.unfold)or ("with_1_43v2ISU" in self.unfold)  )and i==0:            
                    feat=self.to_sf_l2(feat)
                elif ('with_1_4ISU' not in self.unfold) and ('with_1_43ISU' not in self.unfold)and ('with_1_43v2ISU' not in self.unfold):
                    feat=self.to_sf_l2[i](feat)
                i+=1
            if self.quater_nearest!=None:
                if 'both' in self.quater_nearest:
                    rel_coord, q_feat,rel_cell=liif_feat_multiscale_train_quater(feat,coord,scale,self.local_ensemble,self.decode_cell)                
                else:
                    rel_coord, q_feat,rel_cell=liif_feat_multiscale_train(feat,coord,scale,self.local_ensemble,self.decode_cell)        # [B,H*W,2]  [B,H*W,C]        
            else:
                rel_coord, q_feat,rel_cell=liif_feat_multiscale_train(feat,coord,scale,self.local_ensemble,self.decode_cell)        # [B,H*W,2]  [B,H*W,C]        
            if self.pos_enconding: 
                rel_coord = self.pos_encoding(rel_coord) #Position Encoding 
            elif self.pos_enconding_new:
                _,rel_coord = self.pos_encoding(rel_coord) #Position Encoding 
            feat=torch.cat([q_feat,rel_coord],dim=-1)    #[B,H*W,C]
            del rel_coord,q_feat
            if self.decode_cell:
                feat=torch.cat([feat,rel_cell],dim=-1)    #[B,H*W,C]
            latent.append(feat)
        
        latent=torch.cat(latent,dim=-1).view(b*q,-1)    
        latent=self.imnet(latent).view(b,q,self.outputdim)   #[B,H*W,C]
        latent=latent.permute(0,2,1) 
        return latent   #[B,H*W,9]




