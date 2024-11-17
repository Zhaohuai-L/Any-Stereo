import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)           #Instance norm 

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
        



def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost

def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume



def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def disparity_regression(x, maxdisp):       # Same with GwcNet.
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):     # 8 96
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        '''
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)     # feature to attention  [B,48+48,H/4,W/4]
        cv = torch.sigmoid(feat_att)*cv                 # to attention  i.e. 0-1
        return cv

def context_upsample(disp_low, up_weights):  # [B,1,H/4,W/4]  [B,9,H,W]  

    b, c, h, w = disp_low.shape
        
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)   #[B,9,H/4,W/4]  Select the nearest 9 pixels through the unfold    
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)     
        
    return disp





class LayerNormFunction(torch.autograd.Function):

    @staticmethod             
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):     
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None



class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))   
                                                                                
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):                    
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
    
class HighRes_Aggregation(nn.Module):

    def __init__(self, input_dim,output_dim ):
        super(HighRes_Aggregation, self).__init__()
        self.embeding=nn.Sequential(
                nn.PixelUnshuffle(2),
                BasicConv_IN(input_dim*4, output_dim, kernel_size=3, stride=1, padding=1),
                )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.head=nn.Sequential(nn.Conv2d(output_dim, output_dim, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(output_dim), nn.ReLU())
    def forward(self, x):   #[B,3,H,W]
        x = self.embeding(x)       #[B,12,H/2,W/2]
        x=x*self.sca(x)             #
        x=self.head(x)             #[B,24,H/2,W/2]   
        return x                    #[B,36,H/4,W/4] 


class HighRes_Aggregation_LN(nn.Module):

    def __init__(self, input_dim,output_dim ):
        super(HighRes_Aggregation_LN, self).__init__()
        self.embeding=nn.Sequential(
                nn.PixelUnshuffle(2),
                BasicConv_IN(input_dim*4, output_dim, kernel_size=3, stride=1, padding=1),
                )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.head=nn.Sequential(nn.Conv2d(output_dim, output_dim, 3, 1, 1, bias=False),
                LayerNorm2d(output_dim), nn.ReLU())
    def forward(self, x):   #[B,3,H,W]
        x = self.embeding(x)       #[B,12,H/2,W/2]
        x=x*self.sca(x)             #
        x=self.head(x)             #[B,24,H/2,W/2]   
        return x                    #[B,36,H/4,W/4] 

class HighRes_Aggregation_LN_GeLU(nn.Module):

    def __init__(self, input_dim,output_dim ):
        super(HighRes_Aggregation_LN_GeLU, self).__init__()
        self.embeding=nn.Sequential(
                nn.PixelUnshuffle(2),
                BasicConv_IN(input_dim*4, output_dim, kernel_size=3, stride=1, padding=1),
                )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.head=nn.Sequential(nn.Conv2d(output_dim, output_dim, 3, 1, 1, bias=False),
                LayerNorm2d(output_dim), nn.GELU())
    def forward(self, x):   #[B,3,H,W]
        x = self.embeding(x)       #[B,12,H/2,W/2]
        x=x*self.sca(x)             #
        x=self.head(x)             #[B,24,H/2,W/2]   
        return x                    #[B,36,H/4,W/4] 
    
    
def context_upsample_multiscale_train(disp_low, up_weights,hr_coord):  # [B,1,H/4,W/4]  [B,9,H,W]  
    b, c, h, w = disp_low.shape
    
    hr_coord.clamp_(-1+1e-6, 1-1e-6) 
    hr_coord=hr_coord.flip(-1).unsqueeze(1)     # [B,1,H*W,2]       
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)     
    disp_unfold = F.grid_sample(disp_unfold, hr_coord,mode='nearest',align_corners=False)[:,:,0,:]    # [B,1,H*W]
    
    


    disp = (disp_unfold*up_weights).sum(1)      #multiply the weights to regress.
        
    return disp  #[B,H*W]


def context_upsample_multiscale_train_quaterp(disp_low, up_weights,hr_coord):  # [B,1,H/4,W/4]  [B,9,H,W]  




    b, c, h, w = disp_low.shape
    vx_lst = [-1, 1]            
    vy_lst = [-1, 1]
    rx = 2 / disp_low.shape[-2] / 2     # 1/W           
    ry = 2 / disp_low.shape[-1] / 2     # 1/H 
    eps_shift = 1e-6
    disps=[]
    for vx in vx_lst:
        for vy in vy_lst:
            coords_= hr_coord.clone() 
            coords_[:,:,0] += vx * rx + eps_shift
            coords_[:,:,1] += vy * ry + eps_shift
            coords_.clamp_(-1+1e-6, 1-1e-6)
            coords_=coords_.flip(-1).unsqueeze(1)     # [B,1,H*W,2]     
            disp = F.grid_sample(disp_low, coords_,mode='nearest',align_corners=False)[:,:,0,:]    # [B,1,H*W]     
            disps.append(disp)

    disps=torch.cat(disps,dim=1) # [B,4,H*W]

    disp = (disps*up_weights).sum(1)   
        
    return disp  #[B,H*W]