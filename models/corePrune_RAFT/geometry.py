import torch
import torch.nn.functional as F
from models.corePrune_RAFT.utils.utils import bilinear_sampler

import time
class CorrBlock1D:
    def __init__(self, init_fmap1, init_fmap2, num_levels=2, radius=4,mask_invalid=False):
        self.num_levels = num_levels
        self.radius = radius
        self.init_corr_pyramid = []
        init_corr = CorrBlock1D.corr(init_fmap1, init_fmap2,mask_invalid)

        b, h, w, _, w2 = init_corr.shape

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):                  
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)




    def __call__(self, disp, coords):           
        r = self.radius             
        b, _, h, w = disp.shape
        out_pyramid = []            
        for i in range(self.num_levels):
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)     
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)   #[B*H*W1,1,9,1]
            
            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)     #[1,80,184,9]
            out_pyramid.append(init_corr)
            
        out = torch.cat(out_pyramid, dim=-1)    #[B,H,W,4*9] [1,80,184,36]
        return out.permute(0, 3, 1, 2).contiguous().float() #[B,162,H,W]
    
    
    @staticmethod
    def corr(fmap1, fmap2,mask_invalid=False):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        if mask_invalid:            
            corr== torch.tril(corr)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr