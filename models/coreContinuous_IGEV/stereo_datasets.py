import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
import cv2
from models.coreContinuous_IGEV.utils import frame_utils
from models.coreContinuous_IGEV.utils.augmentor import FlowAugmentor, SparseFlowAugmentor,FlowAugmentorWoCrop,SparseFlowAugmentorWoCrop

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):          
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)            
        seq = v0 + r + (2 * r) * torch.arange(n).float()    # (1-n)/n ~ (n-1)/n
        coord_seqs.append(seq) # [(1-H)/H ~ (H-1)/H]  [(1-W)/W ~ (W-1)/W]
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)     
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])  # [H*W,2(H,W)]  
    rgb = img.view(1, -1).permute(1, 0)       # [H*W,1]       
    return coord, rgb

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None,multi_training=False,multi_input_training=False,scale_min=1,scale_max=4,inp_size=[132,240],without_mutli_scale=False):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                if (multi_training==True) and (without_mutli_scale==False):
                    self.augmentor = SparseFlowAugmentorWoCrop(**aug_params)
                else:
                    self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                if multi_training:
                    self.augmentor = FlowAugmentorWoCrop(**aug_params)
                else:
                    self.augmentor = FlowAugmentor(**aug_params)
                
        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.scale_min=scale_min
        self.scale_max=scale_max
        self.multi_training=multi_training
        self.multi_input_training=multi_input_training        
        self.without_mutli_scale=without_mutli_scale
        self.inp_size=inp_size
        self.sample_q=inp_size[0]*inp_size[1]       
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512     

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)
        #Multi-Scale Training Getting random scale
        if (self.multi_training==True):         
            if self.without_mutli_scale == False:
                if self.scale_min!=self.scale_max:
                    scale=random.uniform(self.scale_min, self.scale_max)            
                else:
                    scale=self.scale_max
                h_lr=self.inp_size[0]
                w_lr=self.inp_size[1]
                h_hr=round(h_lr*scale)
                w_hr=round(w_lr*scale)    
            else:
                scale=1
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                if (self.multi_training==True) and (self.without_mutli_scale==False):
                    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid,crop_size=[h_hr,w_hr],scale_size=[h_lr,w_lr])
                else:                
                    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:

                if self.multi_training:
                    img1, img2, flow = self.augmentor(img1, img2, flow,crop_size=[h_hr,w_hr],scale_size=[h_lr,w_lr])
                    #[h_lr,w_lr,3] [h_lr,w_lr,3] [h_hr,w_hr,2]
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)
        
        flow = flow[:,:,:1]         #[1,h_lr,w_lr]
        if self.without_mutli_scale==True:
            h_lr,w_lr,_=flow.shape
        flow_low_res = cv2.resize(flow, dsize=[w_lr//4,h_lr//4], interpolation=cv2.INTER_LINEAR)
        flow_low_res=flow_low_res/(4.*scale)
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() #[c,h_lr,w_lr]
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow_low_res = torch.from_numpy(flow_low_res).float()

        

            
        if self.img_pad is not None:  
            assert False
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)
        if self.multi_training:         
            hr_coord, hr_flow = to_pixel_samples(flow)   
            arry_scale=torch.tensor([scale])
            if self.without_mutli_scale==False:
                if self.sparse:      
                    valid=hr_flow>0.
                    valid_hr_flow=hr_flow[valid].unsqueeze(1)
                    valid_hr_coord=torch.cat((hr_coord[:,:1][valid].unsqueeze(1),hr_coord[:,1:2][valid].unsqueeze(1)),1)
                    if self.sample_q<len(valid_hr_coord):
                        sample_lst = np.random.choice(
                            len(valid_hr_coord), self.sample_q, replace=False)      
                        valid_hr_coord = valid_hr_coord[sample_lst] #[H*W,2]
                        valid_hr_flow = valid_hr_flow[sample_lst]   #[H*W,1]                         
                    else: 
                        invalid_hr_flow=hr_flow[~valid].unsqueeze(1)
                        invalid_hr_coord=torch.cat((hr_coord[:,:1][~valid].unsqueeze(1),hr_coord[:,1:2][~valid].unsqueeze(1)),1)
                        sample_lst = np.random.choice(
                            len(invalid_hr_coord), self.sample_q-len(valid_hr_coord), replace=False)                      
                        invalid_hr_coord = invalid_hr_coord[sample_lst] #[H*W,2]       
                        invalid_hr_flow = invalid_hr_flow[sample_lst]   #[H*W,1]       
                        valid_hr_flow=torch.cat((valid_hr_flow,invalid_hr_flow),0)                             
                        valid_hr_coord=torch.cat((valid_hr_coord,invalid_hr_coord),0)       
                    return self.image_list[index] + [self.disparity_list[index]],img1,img2,valid_hr_coord,valid_hr_flow.permute(1,0),arry_scale                                 
                else:
                    sample_lst = np.random.choice(
                        len(hr_coord), self.sample_q, replace=False)    
                    hr_coord = hr_coord[sample_lst] #[H*W,2]
                    hr_flow = hr_flow[sample_lst]   #[H*W,1]                
            else:
                if self.sparse:
                    valid=hr_flow>0.
                    valid_hr_flow=hr_flow[valid].unsqueeze(1)
                    valid_hr_coord=torch.cat((hr_coord[:,:1][valid].unsqueeze(1),hr_coord[:,1:2][valid].unsqueeze(1)),1)   
                    if self.sample_q<len(valid_hr_coord):
                        print("sample_q is {} valid is {}".format(self.sample_q,len(valid_hr_coord)))
                        assert False ,"Note sample_q is too small, cannot include all valid pixels"                        
                    else:                     
                        invalid_hr_flow=hr_flow[~valid].unsqueeze(1)
                        invalid_hr_coord=torch.cat((hr_coord[:,:1][~valid].unsqueeze(1),hr_coord[:,1:2][~valid].unsqueeze(1)),1)
         
                        invalid_hr_coord = invalid_hr_coord[:self.sample_q-len(valid_hr_coord)] #[H*W,2]    
                        invalid_hr_flow = invalid_hr_flow[:self.sample_q-len(valid_hr_coord)]   #[H*W,1]    
                                                        
                        valid_hr_flow=torch.cat((valid_hr_flow,invalid_hr_flow),0)                             
                        valid_hr_coord=torch.cat((valid_hr_coord,invalid_hr_coord),0)  
                    return self.image_list[index] + [self.disparity_list[index]],img1,img2,valid_hr_coord,valid_hr_flow.permute(1,0),arry_scale,flow_low_res                                                                                                                                                                                                                                  
            return self.image_list[index] + [self.disparity_list[index]],img1,img2,hr_coord,hr_flow.permute(1,0),arry_scale,flow_low_res
        elif self.multi_input_training:  
            scale=random.uniform(self.scale_min, self.scale_max) 
            hwant_hr,wwant_hr=img1.shape[-2:]
            w_lr=int(math.ceil((wwant_hr/scale)))
            h_lr=int(math.ceil((hwant_hr/scale)))
            img1=F.interpolate(img1.unsqueeze(0),(h_lr,w_lr),mode='bicubic',align_corners=False).squeeze(0)
            img2=F.interpolate(img2.unsqueeze(0),(h_lr,w_lr),mode='bicubic',align_corners=False).squeeze(0)       
            pad_ht=hwant_hr-h_lr
            pad_wd=wwant_hr-w_lr
            pad=[pad_wd//2,pad_wd-pad_wd//2,pad_ht//2,pad_ht-pad_ht//2]
            img1=F.pad(img1, pad, mode='replicate')
            img2=F.pad(img2, pad, mode='replicate')
            h_hr_pad=int(math.ceil((img1.shape[-2]*scale)))   
            w_hr_pad=int(math.ceil((img1.shape[-1]*scale)))  
            hr_pad_coord=make_coord([h_hr_pad,w_hr_pad],flatten=False)
            pad=[int(math.ceil(i*scale)) for i in pad]
            hr_pad_coord=hr_pad_coord[pad[2]:h_hr_pad-pad[3],pad[0]:w_hr_pad-pad[1],:]                                            
            if hr_pad_coord.shape[0]!=hwant_hr or hr_pad_coord.shape[1]!=wwant_hr :     #[H,W,C]    
                hr_pad_coord=F.interpolate(hr_pad_coord.permute(2,0,1).unsqueeze(0), (hwant_hr,wwant_hr), mode='bilinear').squeeze(0).permute(1,2,0)                        
            hr_pad_coord=hr_pad_coord.contiguous().view(hwant_hr*wwant_hr,-1)
            arry_scale=torch.tensor([scale])
            flow=flow.view(1,-1)
            return self.image_list[index] + [self.disparity_list[index]], img1, img2, hr_pad_coord,flow,arry_scale
        else:    
            return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None,multi_training=False,multi_input_training=False,scale_min=1,scale_max=2,inp_size=[132,240],root='/mnt/FastData/', dstype='frames_finalpass', things_test=False):           #We set the same parameters with IGEV , i.e., frames_finalpass 
        super(SceneFlowDatasets, self).__init__(aug_params,multi_training=multi_training,multi_input_training=multi_input_training,scale_min=scale_min,scale_max=scale_max,inp_size=inp_size) 
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)     
        root = self.root
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )          # Judging by the level of directory
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)   
        val_idxs = set(np.random.permutation(len(left_images)))
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """
        original_length = len(self.disparity_list)
        # root = osp.join(self.root, 'Monkaa')
        root = self.root
        left_images = sorted(  glob(osp.join(root, self.dstype, split,'*/left/*.png')))
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        # root = osp.join(self.root, 'Driving')
        root = self.root
        left_images = sorted( glob(osp.join(root, self.dstype, split,'*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/data//ETH3D/Stereo', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[]):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/data/Kitti/Kitti2015', image_set='training'):
        super(KITTI, self).__init__(aug_params,sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)
        
                
        image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI12(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/data/Kitti/Kitti2012/stereoflow', image_set='training'):
        super(KITTI12, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)
        
                
        image1_list = sorted(glob(os.path.join(root, image_set, 'colored_0/*_10.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'colored_1/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTImixed(StereoDataset): 
    def __init__(self, aug_params=None, multi_training=False,multi_input_training=False,scale_min=1,scale_max=2,inp_size=[132,240],without_mutli_scale=False,root=None, image_set='training',mode='mix_train'):
        super(KITTImixed, self).__init__(aug_params,  multi_training=multi_training,multi_input_training=multi_input_training,scale_min=scale_min,scale_max=scale_max,inp_size=inp_size,without_mutli_scale=without_mutli_scale, sparse=True, reader=frame_utils.readDispKITTI)
        # assert os.path.exists(root)
        
        root_12 = '/mnt/data/Kitti/Kitti2012/stereoflow'
        image1_list_12 = sorted(glob(os.path.join(root_12, 'training', 'colored_0/*_10.png')))
        image2_list_12 = sorted(glob(os.path.join(root_12, 'training', 'colored_1/*_10.png')))
        disp_list_12 = sorted(glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png'))) 

        root_15 = '/mnt/data/Kitti/Kitti2015'
        image1_list_15 = sorted(glob(os.path.join(root_15, 'training', 'image_2/*_10.png')))
        image2_list_15 = sorted(glob(os.path.join(root_15, 'training', 'image_3/*_10.png')))
        disp_list_15 = sorted(glob(os.path.join(root_15, 'training', 'disp_occ_0/*_10.png')))       
        
        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val12_idxs = set(np.random.permutation(len(image1_list_12))[:14])
        val15_idxs = set(np.random.permutation(len(image1_list_15))[:20])        
        np.random.set_state(state)        
          
        if mode=='mix_train':
            for idx, (img1, img2, disp) in enumerate(zip(image1_list_12, image2_list_12, disp_list_12)):
                if idx not in val12_idxs:
                    self.image_list += [ [img1, img2] ]
                    self.disparity_list += [ disp ]
            for idx, (img1, img2, disp) in enumerate(zip(image1_list_15, image2_list_15, disp_list_15)):
                if idx not in val15_idxs:
                    self.image_list += [ [img1, img2] ]
                    self.disparity_list += [ disp ]            
        elif mode=='mix_train_all': 
            for idx, (img1, img2, disp) in enumerate(zip(image1_list_12, image2_list_12, disp_list_12)):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
            for idx, (img1, img2, disp) in enumerate(zip(image1_list_15, image2_list_15, disp_list_15)):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]                            
        elif mode=='valid_15':
            for idx, (img1, img2, disp) in enumerate(zip(image1_list_15, image2_list_15, disp_list_15)):
                if idx in val15_idxs:
                    self.image_list += [ [img1, img2] ]
                    self.disparity_list += [ disp ]                                      
        elif mode=='valid_12':
            for idx, (img1, img2, disp) in enumerate(zip(image1_list_12, image2_list_12, disp_list_12)):
                if idx in val12_idxs:
                    self.image_list += [ [img1, img2] ]
                    self.disparity_list += [ disp ]  
        elif mode=='15_train':
            for idx, (img1, img2, disp) in enumerate(zip(image1_list_15, image2_list_15, disp_list_15)):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]   
        elif mode=='12_train':
            for idx, (img1, img2, disp) in enumerate(zip(image1_list_12, image2_list_12, disp_list_12)):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]                             





class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, multi_training=False,multi_input_training=False,scale_min=1,scale_max=2,inp_size=[132,240],without_mutli_scale=False,root='/home/Middlebury', split='F'):
        super(Middlebury, self).__init__(aug_params,   multi_training=multi_training,multi_input_training=multi_input_training,scale_min=scale_min,scale_max=scale_max,inp_size=inp_size,without_mutli_scale=without_mutli_scale,sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2014","2014Add"]       
        if split == "2014" or split == "2014Add": 
            scenes = list((Path(root) / split).glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:               # changed for full image test.
            lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]

  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size,
                  'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_', ''), multi_training=args.multi_training, multi_input_training=args.multi_input_training,
                                     scale_min=args.scale_min, scale_max=args.scale_max, inp_size=args.inp_size, without_mutli_scale=args.without_mutli_scale)
        elif dataset_name == 'sceneflow':
            final_dataset = SceneFlowDatasets(aug_params, multi_training=args.multi_training, multi_input_training=args.multi_input_training,
                                              scale_min=args.scale_min, scale_max=args.scale_max, inp_size=args.inp_size, dstype='frames_finalpass')
            new_dataset = final_dataset
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif 'kitti' in dataset_name:
            if '15only' in dataset_name:
                new_dataset = KITTImixed(aug_params, mode='15_train', multi_training=args.multi_training, multi_input_training=args.multi_input_training,
                                         scale_min=args.scale_min, scale_max=args.scale_max, inp_size=args.inp_size, without_mutli_scale=args.without_mutli_scale)
            elif '12only' in dataset_name:
                new_dataset = KITTImixed(aug_params, mode='12_train', multi_training=args.multi_training, multi_input_training=args.multi_input_training,
                                         scale_min=args.scale_min, scale_max=args.scale_max, inp_size=args.inp_size, without_mutli_scale=args.without_mutli_scale)
            elif 'all' in dataset_name:
                new_dataset = KITTImixed(aug_params, mode='mix_train_all', multi_training=args.multi_training, multi_input_training=args.multi_input_training,
                                         scale_min=args.scale_min, scale_max=args.scale_max, inp_size=args.inp_size, without_mutli_scale=args.without_mutli_scale)
            else:
                new_dataset = KITTImixed(aug_params, mode='mix_train', multi_training=args.multi_training, multi_input_training=args.multi_input_training,
                                         scale_min=args.scale_min, scale_max=args.scale_max, inp_size=args.inp_size, without_mutli_scale=args.without_mutli_scale)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params)*140
            logging.info(
                f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params)*5
            logging.info(
                f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(
                aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=8, drop_last=True)
    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader
