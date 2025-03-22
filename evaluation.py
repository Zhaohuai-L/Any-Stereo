from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from models.corePrune_RAFT.raft_stereo import  autocast
import models.corePrune_RAFT.stereo_datasets as datasets
import models.corePrune_RAFT.stereo_datasets as datasetsSF
from PIL import Image
from models.corePrune_RAFT.utils import frame_utils
from models.corePrune_RAFT.utils.utils import InputPadder
from torch.utils.data import DataLoader
from models import __models__
from metrics_utils import *   
from torchvision.utils import save_image
from tensorboardX import SummaryWriter  #
from models.corePrune_RAFT.stereo_datasets import make_coord
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import os
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



avg_test_scalars = AverageMeterDict()    
avg_test_scalars_occ = AverageMeterDict()     
avg_test_scalars_nonocc = AverageMeterDict()     
time_stack=[]

def Disp_to_color(Disp,max_disp=192):
    """Disp to color  borrowed from Kitti Tookkit matlab
    Params: Disp [1,H,W]
    Returns:
        ouput [3,H,W]
    """
    map = torch.from_numpy(np.array([[0, 0, 0, 114],
                                     [0, 0, 1, 185],
                                     [1, 0, 0, 114],
                                     [1, 0, 1, 174],
                                     [0, 1, 0, 114],
                                     [0, 1, 1, 185],
                                     [1, 1, 0, 114],
                                     [1, 1, 1, 0]]))

    bins = map[0:-1, 3:4].squeeze(1)
    cbins = torch.cumsum(bins, dim=0)
    bins = bins/cbins[-1]
    cbins = cbins[0:-1] / cbins[-1]
    Dispmin = torch.clamp(Disp/max_disp, 0, 1)
    Disprepeat = Dispmin.repeat(6, 1, 1)
    cbinsrepeat = cbins.unsqueeze(1).unsqueeze(1)
    ind = torch.sum(Disprepeat > cbinsrepeat, dim=0, keepdim=True)
    bins = 1.0/bins
    cbins = torch.cat((torch.tensor([0]), cbins), 0)
    Dispout = (Dispmin-cbins[ind])*bins[ind]
    output = map[ind, :3] * (1-Dispout).unsqueeze(3).repeat(1, 1, 1, 3) + \
        map[ind+1, :3] * (Dispout).unsqueeze(3).repeat(1, 1, 1, 3)
    output = output.squeeze(0).permute(2, 0, 1)

    return output

def pad_for_multi_train(args,image1,image2):
    assert args.scale_test >0.99 
    h_want, w_want = image1.shape[-2:]
    h_lr=int(math.ceil(h_want/float(args.scale_test)))
    w_lr=int(math.ceil(w_want/float(args.scale_test)))  
    if args.scale_test>1:
        image1=F.interpolate(image1,(h_lr,w_lr),mode='bicubic',align_corners=False) 
        image2=F.interpolate(image2,(h_lr,w_lr),mode='bicubic',align_corners=False)
    if 'IGEVStereo' in args.model: 
        padder = InputPadder(image1.shape, divis_by=32)
    else:
        padder = InputPadder(image1.shape, divis_by=16)
    image1_pad, image2_pad = padder.pad(image1, image2) 
    pad_num=padder.get_pad_num()
    h_hr_pad=int((image1_pad.shape[2])*args.scale_test) 
    w_hr_pad=int((image1_pad.shape[3])*args.scale_test) 
    hr_pad_coord=make_coord([h_hr_pad,w_hr_pad],flatten=False)
    pad_num=[int(i*args.scale_test) for i in pad_num] 
    hr_pad_coord=hr_pad_coord[pad_num[0]:h_hr_pad-pad_num[1],pad_num[2]:w_hr_pad-pad_num[3],:]
    if hr_pad_coord.shape[0]!=h_want or hr_pad_coord.shape[1]!=w_want :     #[H,W,C]
        hr_pad_coord=F.interpolate(hr_pad_coord.permute(2,0,1).unsqueeze(0), (h_want,w_want), mode='bilinear').squeeze(0).permute(1,2,0)
    hr_coord=hr_pad_coord.contiguous().view(h_want*w_want,-1)
    return image1_pad,image2_pad,hr_coord

def pad_for_muti_other(args,image1,image2):
    h_want,w_want=image1.shape[-2:] 
    h_lr=math.ceil(h_want/args.scale_test)
    w_lr=math.ceil(w_want/args.scale_test)
    if args.scale_test>1: 
        image1=F.interpolate(image1,(h_lr,w_lr),mode='bicubic',align_corners=False)  
        image2=F.interpolate(image2,(h_lr,w_lr),mode='bicubic',align_corners=False)
    padder = InputPadder(image1.shape, divis_by=32)
    image1_pad, image2_pad = padder.pad(image1, image2)
    return image1_pad,image2_pad,padder


@torch.no_grad()
def validate_eth3d(args,logger,model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()      
    avg_test_scalars.clean(),avg_test_scalars_occ.clean(),avg_test_scalars_nonocc.clean()    
    val_dataset = datasets.ETH3D(aug_params= {})

    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        name=imageL_file.split('/')[-2]; image1 = image1[None].cuda(); image2 = image2[None].cuda()
        if args.multi_training or args.multi_input_training: 
            image1_pad,image2_pad,hr_coord=pad_for_multi_train(args,image1,image2)
        elif args.multi_evaothers:
            image1_pad, image2_pad,padder=pad_for_muti_other(args,image1,image2)            
        else:        
            padder = InputPadder(image1.shape, divis_by=32)
            image1_pad, image2_pad = padder.pad(image1, image2)
        if args.model=='RAFTStereo': 
            with autocast(enabled=mixed_prec):
                start = time.time()
                _, flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                end = time.time()
                if args.multi_evaothers:
                    flow_pr = padder.unpad(flow_pr).cpu() 
                    if args.scale_test>1:
                        flow_pr=F.interpolate(flow_pr*args.scale_test,(image1.shape[2],image1.shape[3]),mode='bicubic',align_corners=False)
                    flow_pr=flow_pr.squeeze(0) 
                    disp=-flow_pr; disp_gt=flow_gt   
                else:
                    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0); disp=-flow_pr; disp_gt= flow_gt    
        else:
            with autocast(enabled=mixed_prec):
                if args.multi_training or args.multi_input_training:
                    flow_pr = model(image1_pad, image2_pad, iters=iters, hr_coord=hr_coord.unsqueeze(0),scale=torch.tensor([args.scale_test]).view(1,1),test_mode=True)
                    flow_pr=flow_pr.view(1,image1.shape[2],image1.shape[3]).cpu();disp=flow_pr; disp_gt=flow_gt
                elif args.multi_evaothers:
                    start = time.time()
                    flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                    end = time.time()
                    flow_pr = padder.unpad(flow_pr).cpu() 
                    if args.scale_test>1:
                        flow_pr=F.interpolate(flow_pr*args.scale_test,(image1.shape[2],image1.shape[3]),mode='bicubic',align_corners=False)
                    flow_pr=flow_pr.squeeze(0)                         
                    disp=flow_pr; disp_gt=flow_gt                      
                else:                
                    flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)               
                    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0); disp=flow_pr; disp_gt=flow_gt
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        disp = torch.where(torch.isinf(disp), torch.full_like(disp, 0), disp)        
        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = torch.from_numpy(np.ascontiguousarray(occ_mask)).unsqueeze(0)
        valid_mask=(valid_gt>=-0.5)&(disp_gt<1000)                 
        if args.max_enable:
            valid_mask=valid_mask&(disp_gt<args.max_disp)
        mask_occlu=valid_mask&(occ_mask==255)
        mask_nonocclu=(~mask_occlu.bool())&valid_mask   
        scalar_outputs = {}
        scalar_outputs_nonocc = {}     
        scalar_outputs_occ = {}
        if float(torch.sum(mask_occlu)) > 0:
            disp_gt=torch.squeeze(disp_gt,1);valid_mask=torch.squeeze(valid_mask,1);disp_ests=[torch.squeeze(disp,1)]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, valid_mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, valid_mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, valid_mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, valid_mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, valid_mask, 3.0) for disp_est in disp_ests]
            mask_occlu=torch.squeeze(mask_occlu,1);mask_nonocclu=torch.squeeze(mask_nonocclu,1)
            scalar_outputs_nonocc["EPE"] = [EPE_metric(disp_est, disp_gt, mask_occlu) for disp_est in disp_ests]
            scalar_outputs_nonocc["D1"] = [D1_metric(disp_est, disp_gt, mask_occlu) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres1"] = [Thres_metric(disp_est, disp_gt, mask_occlu, 1.0) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres2"] = [Thres_metric(disp_est, disp_gt, mask_occlu, 2.0) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres3"] = [Thres_metric(disp_est, disp_gt, mask_occlu, 3.0) for disp_est in disp_ests]
            scalar_outputs_occ["EPE"] = [EPE_metric(disp_est, disp_gt, mask_nonocclu) for disp_est in disp_ests]
            scalar_outputs_occ["D1"] = [D1_metric(disp_est, disp_gt, mask_nonocclu) for disp_est in disp_ests]
            scalar_outputs_occ["Thres1"] = [Thres_metric(disp_est, disp_gt, mask_nonocclu, 1.0) for disp_est in disp_ests]
            scalar_outputs_occ["Thres2"] = [Thres_metric(disp_est, disp_gt, mask_nonocclu, 2.0) for disp_est in disp_ests]
            scalar_outputs_occ["Thres3"] = [Thres_metric(disp_est, disp_gt, mask_nonocclu, 3.0) for disp_est in disp_ests]
        if args.ShowImage:
            mask_nonocclu=mask_nonocclu.unsqueeze(1); mask_occlu=mask_occlu.unsqueeze(1)
            show_mask_occlu=image1.clone().detach()
            show_mask_nonocclu=image1.clone().detach()
            show_mask_occlu=show_mask_occlu*mask_occlu.cuda()
            show_mask_nonocclu=show_mask_nonocclu*mask_nonocclu.cuda()
            D_est=(disp_ests[0].squeeze(1))*valid_mask; D_gt =  (disp_gt.squeeze(1))*valid_mask
            errormap= disp_error_image_func.apply(D_est,D_gt)
            disp=disp_ests[0]*valid_mask   
            disp_gt = torch.where(torch.isinf(disp_gt), torch.full_like(disp_gt, 0), disp_gt)
            image_outputs = {"img_left": image1,"img_right": image2, "disp_gt": disp_gt, "disp_Esti": disp*valid_mask,"error_map":errormap,"mask_occ":show_mask_occlu,"mask_nonocc":show_mask_nonocclu}                   
            save_images(logger, 'Test', image_outputs, val_id)   
        if args.record:
            with open(args.savepath+'/result.txt','a',encoding='utf-8') as f:
                f.write(" {} EPE is = {:.3f} `Occlu = {:.3f}` `NonOcclu = {:.3f}`\n".format(name,scalar_outputs["EPE"][0],scalar_outputs_occ["EPE"][0],scalar_outputs_nonocc["EPE"][0]))              
        if args.output:
            color_disp=Disp_to_color(disp.squeeze(1).cpu())
            save_image(color_disp,os.path.join(args.savepath,"output","disp_{}.png".format(name)))
        print("{} EPE is = {:.3f} ".format(name,scalar_outputs["EPE"][0]))
        avg_test_scalars.update(tensor2float(scalar_outputs))
        avg_test_scalars_occ.update(tensor2float(scalar_outputs_occ))
        avg_test_scalars_nonocc.update(tensor2float(scalar_outputs_nonocc))      
        save_scalars(logger, 'Test', scalar_outputs, val_id)
        save_scalars(logger, 'Test_Non_occlusion_area', scalar_outputs_nonocc, val_id)  
        save_scalars(logger, 'Test_occlusion_area', scalar_outputs_occ, val_id)                      
    return avg_test_scalars,avg_test_scalars_occ,avg_test_scalars_nonocc                               



@torch.no_grad()
def validate_kitti(args,logger,model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()      
    avg_test_scalars.clean(),avg_test_scalars_occ.clean(),avg_test_scalars_nonocc.clean()    
    if args.dataset=='kitti15':
        val_dataset = datasets.KITTI(aug_params= {}, image_set='training')
    elif args.dataset=='kitti12':
        val_dataset = datasets.KITTI12(aug_params= {}, image_set='training')
    else : assert False
    torch.backends.cudnn.benchmark = True
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        name=imageL_file.split('/')[-1].split('.')[-2]; image1 = image1[None].cuda(); image2 = image2[None].cuda()
        
        if args.multi_training or args.multi_input_training: 
            image1_pad,image2_pad,hr_coord=pad_for_multi_train(args,image1,image2)
        elif args.multi_evaothers:
            image1_pad, image2_pad,padder=pad_for_muti_other(args,image1,image2)            
        else:        
            padder = InputPadder(image1.shape, divis_by=32)
            image1_pad, image2_pad = padder.pad(image1, image2)

        if args.model=='RAFTStereo': 
            with autocast(enabled=mixed_prec):
                start = time.time()
                _, flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                end = time.time()
                if args.multi_evaothers:
                    flow_pr = padder.unpad(flow_pr).cpu() 
                    if args.scale_test>1:
                        flow_pr=F.interpolate(flow_pr*args.scale_test,(image1.shape[2],image1.shape[3]),mode='bicubic',align_corners=False)
                    flow_pr=flow_pr.squeeze(0) 
                    disp=-flow_pr; disp_gt=flow_gt 
                else:
                    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0); disp=-flow_pr; disp_gt= flow_gt                                                      
        else:
            with autocast(enabled=mixed_prec):
                if args.multi_training or args.multi_input_training:
                    start = time.time()
                    flow_pr = model(image1_pad, image2_pad, iters=iters, hr_coord=hr_coord.unsqueeze(0),scale=torch.tensor([args.scale_test]).view(1,1),test_mode=True)
                    end = time.time()
                    flow_pr=flow_pr.view(1,image1.shape[2],image1.shape[3]).cpu();disp=flow_pr; disp_gt=flow_gt
                elif args.multi_evaothers:
                    start = time.time()
                    flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                    end = time.time()
                    flow_pr = padder.unpad(flow_pr).cpu() 
                    if args.scale_test>1:
                        flow_pr=F.interpolate(flow_pr*args.scale_test,(image1.shape[2],image1.shape[3]),mode='bicubic',align_corners=False)
                    flow_pr=flow_pr.squeeze(0) 
                    disp=flow_pr; disp_gt=flow_gt                   
                else:
                    start = time.time()
                    flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                    end = time.time()
                    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0); disp=flow_pr; disp_gt=flow_gt
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        if args.dataset=='kitti15':
            non_image = Image.open(GT_file.replace('disp_occ_0', 'disp_noc_0'))
            occ_image = Image.open(GT_file)
        elif args.dataset=='kitti12':
            non_image = Image.open(GT_file.replace('disp_occ', 'disp_noc'))
            occ_image = Image.open(GT_file)
        non_image = torch.from_numpy(np.ascontiguousarray(non_image)).unsqueeze(0)
        occ_image = torch.from_numpy(np.ascontiguousarray(occ_image)).unsqueeze(0)
        valid_mask=(valid_gt>=0.5)&(disp_gt<1000)                 
        if args.max_enable:
            valid_mask=valid_mask&(disp_gt<args.max_disp)
        mask_occlu=valid_mask&(occ_image==non_image)
        mask_nonocclu=(~mask_occlu.bool())&valid_mask   
        scalar_outputs = {}
        scalar_outputs_nonocc = {}     
        scalar_outputs_occ = {}
        if float(torch.sum(mask_occlu)) > 0:
            disp_gt=torch.squeeze(disp_gt,1);valid_mask=torch.squeeze(valid_mask,1);disp_ests=[torch.squeeze(disp,1)]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, valid_mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, valid_mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, valid_mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, valid_mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, valid_mask, 3.0) for disp_est in disp_ests]
            mask_occlu=torch.squeeze(mask_occlu,1);mask_nonocclu=torch.squeeze(mask_nonocclu,1)
            scalar_outputs_nonocc["EPE"] = [EPE_metric(disp_est, disp_gt, mask_occlu) for disp_est in disp_ests]
            scalar_outputs_nonocc["D1"] = [D1_metric(disp_est, disp_gt, mask_occlu) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres1"] = [Thres_metric(disp_est, disp_gt, mask_occlu, 1.0) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres2"] = [Thres_metric(disp_est, disp_gt, mask_occlu, 2.0) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres3"] = [Thres_metric(disp_est, disp_gt, mask_occlu, 3.0) for disp_est in disp_ests]
            if float(torch.sum(mask_nonocclu)) > 0:
                scalar_outputs_occ["EPE"] = [EPE_metric(disp_est, disp_gt, mask_nonocclu) for disp_est in disp_ests]
                scalar_outputs_occ["D1"] = [D1_metric(disp_est, disp_gt, mask_nonocclu) for disp_est in disp_ests]
                scalar_outputs_occ["Thres1"] = [Thres_metric(disp_est, disp_gt, mask_nonocclu, 1.0) for disp_est in disp_ests]
                scalar_outputs_occ["Thres2"] = [Thres_metric(disp_est, disp_gt, mask_nonocclu, 2.0) for disp_est in disp_ests]
                scalar_outputs_occ["Thres3"] = [Thres_metric(disp_est, disp_gt, mask_nonocclu, 3.0) for disp_est in disp_ests]
        if args.ShowImage:
            mask_nonocclu=mask_nonocclu.unsqueeze(1); mask_occlu=mask_occlu.unsqueeze(1)
            show_mask_occlu=image1.clone().detach()
            show_mask_nonocclu=image1.clone().detach()
            show_mask_occlu=show_mask_occlu*mask_occlu.cuda()
            show_mask_nonocclu=show_mask_nonocclu*mask_nonocclu.cuda()
            D_est=(disp_ests[0].squeeze(1))*valid_mask; D_gt =  (disp_gt.squeeze(1))*valid_mask 
            errormap= disp_error_image_func.apply(D_est,D_gt)
            disp=disp_ests[0]*valid_mask   
            disp_gt = torch.where(torch.isinf(disp_gt), torch.full_like(disp_gt, 0), disp_gt)
            image_outputs = {"img_left": image1,"img_right": image2, "disp_gt": disp_gt, "disp_Esti": disp*valid_mask,"error_map":errormap,"mask_occ":show_mask_occlu,"mask_nonocc":show_mask_nonocclu}                   
            save_images(logger, 'Test', image_outputs, val_id)   
        if args.record:
            with open(args.savepath+'/result.txt','a',encoding='utf-8') as f:
                f.write(" {} EPE is = {:.3f} `Occlu = {:.3f}` `NonOcclu = {:.3f}`  time= {:.3f}\n".format(name,scalar_outputs["EPE"][0],scalar_outputs_occ["EPE"][0] if "EPE" in scalar_outputs_occ else 0,scalar_outputs_nonocc["EPE"][0],end-start))              
        if args.output:
            color_disp=Disp_to_color(disp.squeeze(1).cpu())
            save_image(color_disp,os.path.join(args.savepath,"output","disp_{}.png".format(name)))
        print("{} EPE is = {:.3f} time is {:.3f}".format(name,scalar_outputs["EPE"][0],end-start))
        avg_test_scalars.update(tensor2float(scalar_outputs))
        avg_test_scalars_occ.update(tensor2float(scalar_outputs_occ))
        avg_test_scalars_nonocc.update(tensor2float(scalar_outputs_nonocc))      
        save_scalars(logger, 'Test', scalar_outputs, val_id)
        save_scalars(logger, 'Test_Non_occlusion_area', scalar_outputs_nonocc, val_id)  
        save_scalars(logger, 'Test_occlusion_area', scalar_outputs_occ, val_id)       
        
        time_stack.append(end-start)                            
    return avg_test_scalars,avg_test_scalars_occ,avg_test_scalars_nonocc,time_stack


@torch.no_grad()
def validate_things(args,logger,model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()  
    avg_test_scalars.clean(),avg_test_scalars_occ.clean(),avg_test_scalars_nonocc.clean()    
    val_dataset = datasetsSF.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)  
    test_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
    pin_memory=True, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)

    testbar=tqdm(test_loader)
    for val_id , data in enumerate(testbar):
        
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, flowright,valid_gt = data
        image1 = image1.cuda(); image2 = image2.cuda(); name="Batch{}".format(val_id);
        bs = image1.shape[0]
        if args.multi_training or args.multi_input_training: 
            image1_pad,image2_pad,hr_coord=pad_for_multi_train(args,image1,image2)
        elif args.multi_evaothers:
            image1_pad, image2_pad,padder=pad_for_muti_other(args,image1,image2)            
        else:        
            padder = InputPadder(image1.shape, divis_by=32)
            image1_pad, image2_pad = padder.pad(image1, image2)            
        if 'RAFTStereo' in args.model: 
            with autocast(enabled=mixed_prec):
                start = time.time()
                _, flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                end = time.time()
                if args.multi_evaothers:
                    flow_pr = padder.unpad(flow_pr).cpu() 
                    if args.scale_test>1:
                        flow_pr=F.interpolate(flow_pr*args.scale_test,(image1.shape[2],image1.shape[3]),mode='bicubic',align_corners=False)
                    flow_pr=flow_pr.squeeze(0) 
                    disp=-flow_pr; disp_gt=flow_gt      
                else:
                    flow_pr = padder.unpad(flow_pr).cpu(); disp=-flow_pr; disp_gt= flow_gt     
        else:
            with autocast(enabled=mixed_prec):
                if args.multi_training or args.multi_input_training:        
                    start = time.time()
                    flow_pr = model(image1_pad, image2_pad, iters=iters, hr_coord=hr_coord.unsqueeze(0).expand(bs, *hr_coord.shape),scale=torch.tensor([args.scale_test]).view(1,1).expand(bs, 1),test_mode=True)
                    end = time.time()
                    flow_pr=flow_pr.view(bs,1,image1.shape[2],image1.shape[3]).cpu();disp=flow_pr; disp_gt=flow_gt
                elif args.multi_evaothers:
                    start = time.time()
                    flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                    end = time.time()
                    flow_pr = padder.unpad(flow_pr).cpu() 
                    if args.scale_test>1:
                        flow_pr=F.interpolate(flow_pr*args.scale_test,(image1.shape[2],image1.shape[3]),mode='bicubic',align_corners=False)
                    flow_pr=flow_pr.squeeze(0)                         
                    disp=flow_pr; disp_gt=flow_gt                  
                else:                
                    start = time.time()
                    flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                    end = time.time()
                    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0); disp=flow_pr; disp_gt=flow_gt
        assert flow_pr.shape == disp_gt.shape, (flow_pr.shape, disp_gt.shape)
        disp = torch.where(torch.isinf(disp), torch.full_like(disp, 0), disp)
        
        valid_mask=(valid_gt.unsqueeze(1)>0.5)&(disp_gt<1000)                 
        if args.max_enable:
            valid_mask=valid_mask&(disp_gt<args.max_disp)  
        mask_occlu=valid_mask&(occ_mask(disp_gt,flowright).bool())
        mask_nonocclu=(~mask_occlu.bool())&valid_mask      
        scalar_outputs = {}
        scalar_outputs_nonocc = {}     
        scalar_outputs_occ = {}
        if float(torch.sum(mask_occlu)) > 0:
            disp_gt=torch.squeeze(disp_gt,1);valid_mask=torch.squeeze(valid_mask,1);disp_ests=[torch.squeeze(disp,1)]
            scalar_outputs["EPE"] = [EPE_metric_filter(disp_est, disp_gt, valid_mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric_filter(disp_est, disp_gt, valid_mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric_filter(disp_est, disp_gt, valid_mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric_filter(disp_est, disp_gt, valid_mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric_filter(disp_est, disp_gt, valid_mask, 3.0) for disp_est in disp_ests]
            mask_occlu=torch.squeeze(mask_occlu,1);mask_nonocclu=torch.squeeze(mask_nonocclu,1)
            scalar_outputs_nonocc["EPE"] = [EPE_metric_filter(disp_est, disp_gt, mask_occlu) for disp_est in disp_ests]
            scalar_outputs_nonocc["D1"] = [D1_metric_filter(disp_est, disp_gt, mask_occlu) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres1"] = [Thres_metric_filter(disp_est, disp_gt, mask_occlu, 1.0) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres2"] = [Thres_metric_filter(disp_est, disp_gt, mask_occlu, 2.0) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres3"] = [Thres_metric_filter(disp_est, disp_gt, mask_occlu, 3.0) for disp_est in disp_ests]
            if float(torch.sum(mask_nonocclu)) > 0:
                scalar_outputs_occ["EPE"] = [EPE_metric_filter(disp_est, disp_gt, mask_nonocclu) for disp_est in disp_ests]
                scalar_outputs_occ["D1"] = [D1_metric_filter(disp_est, disp_gt, mask_nonocclu) for disp_est in disp_ests]
                scalar_outputs_occ["Thres1"] = [Thres_metric_filter(disp_est, disp_gt, mask_nonocclu, 1.0) for disp_est in disp_ests]
                scalar_outputs_occ["Thres2"] = [Thres_metric_filter(disp_est, disp_gt, mask_nonocclu, 2.0) for disp_est in disp_ests]
                scalar_outputs_occ["Thres3"] = [Thres_metric_filter(disp_est, disp_gt, mask_nonocclu, 3.0) for disp_est in disp_ests]
        if args.ShowImage:
            mask_nonocclu=mask_nonocclu.unsqueeze(1); mask_occlu=mask_occlu.unsqueeze(1)
            show_mask_occlu=image1.clone().detach()
            show_mask_nonocclu=image1.clone().detach()
            show_mask_occlu=show_mask_occlu*mask_occlu.cuda()
            show_mask_nonocclu=show_mask_nonocclu*mask_nonocclu.cuda()
            D_est=(disp_ests[0].squeeze(1)[:1])*valid_mask[:1]; D_gt =  (disp_gt[:1].squeeze(1))*valid_mask[:1] 
            errormap= disp_error_image_func.apply(D_est,D_gt)
            disp=disp_ests[0]*valid_mask[0]
            disp_gt = torch.where(torch.isinf(disp_gt), torch.full_like(disp_gt, 0), disp_gt)
            image_outputs = {"img_left": image1,"img_right": image2, "disp_gt": disp_gt, "disp_Esti": disp*valid_mask,"error_map":errormap,"mask_occ":show_mask_occlu,"mask_nonocc":show_mask_nonocclu}                   
            save_images(logger, 'Test', image_outputs, val_id)   
        if args.record: 
            with open(args.savepath+'/result.txt','a',encoding='utf-8') as f:
                f.write(" {} EPE is = {:.3f} `Occlu = {:.3f}` `NonOcclu = {:.3f}`\n".format(name,scalar_outputs["EPE"][0],scalar_outputs_occ["EPE"][0],scalar_outputs_nonocc["EPE"][0]))              
        if args.output:
            color_disp=Disp_to_color(disp[:1].squeeze(1).cpu())
            save_image(color_disp,os.path.join(args.savepath,"output","disp_{}.png".format(name)))
        testbar.set_description("EPE = {:.3f} time: {:.3f} ".format(scalar_outputs["EPE"][0],end-start))      
        time_stack1=time_stack[102:]
        print("average time is {}".format(sum(time_stack1)/len(time_stack1)))
        break      
        avg_test_scalars.update(tensor2float(scalar_outputs))
        avg_test_scalars_occ.update(tensor2float(scalar_outputs_occ))
        avg_test_scalars_nonocc.update(tensor2float(scalar_outputs_nonocc))      
        save_scalars(logger, 'Test', scalar_outputs, val_id)
        save_scalars(logger, 'Test_Non_occlusion_area', scalar_outputs_nonocc, val_id)  
        save_scalars(logger, 'Test_occlusion_area', scalar_outputs_occ, val_id)                                 
    return avg_test_scalars,avg_test_scalars_occ,avg_test_scalars_nonocc                    




@torch.no_grad()
def validate_middlebury(args,logger,model, iters=32, split='F', mixed_prec=False, Before_GRU=False):
    """ Peform validation using the Middlebury-V3 dataset """
    avg_test_scalars.clean(),avg_test_scalars_occ.clean(),avg_test_scalars_nonocc.clean()
    model.eval()   
    val_dataset = datasets.Middlebury(aug_params={}, split=split)
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        name=imageL_file.split('/')[-2]; image1 = image1[None].cuda(); image2 = image2[None].cuda()
        if args.multi_training or args.multi_input_training: 
            image1_pad,image2_pad,hr_coord=pad_for_multi_train(args,image1,image2)
        elif args.multi_evaothers:
            image1_pad, image2_pad,padder=pad_for_muti_other(args,image1,image2)            
        else:        
            padder = InputPadder(image1.shape, divis_by=32)
            image1_pad, image2_pad = padder.pad(image1, image2)
        if args.model=='RAFTStereo': 
            with autocast(enabled=mixed_prec):
                start = time.time()
                _, flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                end = time.time()
                if args.multi_evaothers:
                    flow_pr = padder.unpad(flow_pr).cpu() 
                    if args.scale_test>1:
                        flow_pr=F.interpolate(flow_pr*args.scale_test,(image1.shape[2],image1.shape[3]),mode='bicubic',align_corners=False)
                    flow_pr=flow_pr.squeeze(0) 
                    disp=-flow_pr; disp_gt=flow_gt   
                else:
                    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0); disp=-flow_pr; disp_gt= flow_gt                                      
        else: 
            with autocast(enabled=mixed_prec):
                if args.multi_training or args.multi_input_training:
                    flow_pr = model(image1_pad, image2_pad, iters=iters, hr_coord=hr_coord.unsqueeze(0),scale=torch.tensor([args.scale_test]).view(1,1),test_mode=True)
                    flow_pr=flow_pr.view(1,image1.shape[2],image1.shape[3]).cpu();disp=flow_pr; disp_gt=flow_gt
                elif args.multi_evaothers:
                    flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                    flow_pr = padder.unpad(flow_pr).cpu() 
                    if args.scale_test>1:
                        flow_pr=F.interpolate(flow_pr*args.scale_test,(image1.shape[2],image1.shape[3]),mode='bicubic',align_corners=False)
                    flow_pr=flow_pr.squeeze(0)                         
                    disp=flow_pr; disp_gt=flow_gt                  
                else:                
                    flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True)
                    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0); disp=flow_pr; disp_gt=flow_gt
            
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        disp = torch.where(torch.isinf(disp), torch.full_like(disp, 0), disp)
        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = torch.from_numpy(np.ascontiguousarray(occ_mask, dtype=np.float32)).unsqueeze(0)
        valid_mask=(valid_gt>=-0.5)&(disp_gt<1000)                 
        if args.max_enable:
            valid_mask=valid_mask&(disp_gt<args.max_disp)
        mask_occlu=valid_mask&(occ_mask==255)
        mask_nonocclu=(~mask_occlu.bool())&valid_mask      
        scalar_outputs = {}
        scalar_outputs_nonocc = {}     
        scalar_outputs_occ = {}
        if float(torch.sum(mask_occlu)) > 0:
            disp_gt=torch.squeeze(disp_gt,1);valid_mask=torch.squeeze(valid_mask,1);disp_ests=[torch.squeeze(disp,1)]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, valid_mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, valid_mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, valid_mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, valid_mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, valid_mask, 3.0) for disp_est in disp_ests]
            mask_occlu=torch.squeeze(mask_occlu,1);mask_nonocclu=torch.squeeze(mask_nonocclu,1)
            scalar_outputs_nonocc["EPE"] = [EPE_metric(disp_est, disp_gt, mask_occlu) for disp_est in disp_ests]
            scalar_outputs_nonocc["D1"] = [D1_metric(disp_est, disp_gt, mask_occlu) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres1"] = [Thres_metric(disp_est, disp_gt, mask_occlu, 1.0) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres2"] = [Thres_metric(disp_est, disp_gt, mask_occlu, 2.0) for disp_est in disp_ests]
            scalar_outputs_nonocc["Thres3"] = [Thres_metric(disp_est, disp_gt, mask_occlu, 3.0) for disp_est in disp_ests]
            scalar_outputs_occ["EPE"] = [EPE_metric(disp_est, disp_gt, mask_nonocclu) for disp_est in disp_ests]
            scalar_outputs_occ["D1"] = [D1_metric(disp_est, disp_gt, mask_nonocclu) for disp_est in disp_ests]
            scalar_outputs_occ["Thres1"] = [Thres_metric(disp_est, disp_gt, mask_nonocclu, 1.0) for disp_est in disp_ests]
            scalar_outputs_occ["Thres2"] = [Thres_metric(disp_est, disp_gt, mask_nonocclu, 2.0) for disp_est in disp_ests]
            scalar_outputs_occ["Thres3"] = [Thres_metric(disp_est, disp_gt, mask_nonocclu, 3.0) for disp_est in disp_ests]
        if args.ShowImage:
            mask_nonocclu=mask_nonocclu.unsqueeze(1); mask_occlu=mask_occlu.unsqueeze(1)
            show_mask_occlu=image1.clone().detach()
            show_mask_nonocclu=image1.clone().detach()
            show_mask_occlu=show_mask_occlu*mask_occlu.cuda()
            show_mask_nonocclu=show_mask_nonocclu*mask_nonocclu.cuda()
            D_est=(disp_ests[0].squeeze(1))*valid_mask; D_gt =  (disp_gt.squeeze(1))*valid_mask  
            errormap= disp_error_image_func.apply(D_est,D_gt)
            disp=disp_ests[0]*valid_mask 
            disp_gt = torch.where(torch.isinf(disp_gt), torch.full_like(disp_gt, 0), disp_gt)
            image_outputs = {"img_left": image1,"img_right": image2, "disp_gt": disp_gt, "disp_Esti": disp*valid_mask,"error_map":errormap,"mask_occ":show_mask_occlu,"mask_nonocc":show_mask_nonocclu}                   
            if logger!=None:
                save_images(logger, 'Test', image_outputs, val_id)   
        if args.record:
            with open(args.savepath+'/result.txt','a',encoding='utf-8') as f:
                f.write(" {} EPE is = {:.3f} `Occlu = {:.3f}` `NonOcclu = {:.3f}`\n".format(name,scalar_outputs["EPE"][0],scalar_outputs_occ["EPE"][0],scalar_outputs_nonocc["EPE"][0]))              
        if args.output:
            color_disp=Disp_to_color(disp_gt.squeeze(1).cpu(),max_disp=400)
            save_image(color_disp,os.path.join(args.savepath,"output","disp_{}.png".format(name,int(args.scale_test*10))))
        print("{} EPE is = {:.3f} ".format(name,scalar_outputs["EPE"][0]))
        avg_test_scalars.update(tensor2float(scalar_outputs))
        avg_test_scalars_occ.update(tensor2float(scalar_outputs_occ))
        avg_test_scalars_nonocc.update(tensor2float(scalar_outputs_nonocc))   
        if logger!=None:   
            save_scalars(logger, 'Test', scalar_outputs, val_id)
            save_scalars(logger, 'Test_Non_occlusion_area', scalar_outputs_nonocc, val_id)  
            save_scalars(logger, 'Test_occlusion_area', scalar_outputs_occ, val_id)                                   
    return avg_test_scalars,avg_test_scalars_occ,avg_test_scalars_nonocc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', action='store_true',default=False, help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    parser.add_argument('--max_enable', help='enable the max constraint', action='store_true', default=False) 
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--model', type=str, default='PruneStereo')     
    parser.add_argument('--dataset', help="dataset for evaluation", default='things', choices=["eth3d", "kitti15","kitti12", "things"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--record', help='record on text', action='store_true', default=False) 
    parser.add_argument('--savepath', default='./evaluation/', help='save path')
    parser.add_argument('--ShowImage', help='Show IMage', action='store_true', default=False)
    parser.add_argument('--output', help='Ouput the image as colored png', action='store_true',default=False)        
    parser.add_argument('--local_ensemble', default=False, action='store_true', help='')      
    parser.add_argument('--decode_cell', default=False, action='store_true', help='')    
    parser.add_argument('--unfold', default=False, action='store_true', help='')     
    parser.add_argument('--Raw_Mask_dim', type=int, default=32, help="width of the correlation pyramid")
    parser.add_argument('--pos_enconding_new', default=False, action='store_true', help='')    
    parser.add_argument('--pos_enconding', default=False, action='store_true', help='')    
    parser.add_argument('--require_grad', default=True, action='store_true', help='')    
    parser.add_argument('--pos_dim', type=int, default=0, help="width of the correlation pyramid")
    parser.add_argument('--mlphidden_list', type=int, nargs='+', default=[128]*3, help="size of the random image crops used during training.")
    
    parser.add_argument('--multi_training', default=False, action='store_true', help='')     
    parser.add_argument('--scale_test',type=float, default=1, help="the scale number for test, must lager than 1")    
    
    parser.add_argument('--multi_input_training', default=False, action='store_true', help='')   

    parser.add_argument('--disparity_norm', default=False, action='store_true', help='')   
    parser.add_argument('--disparity_norm2', default=False, action='store_true', help='')     
    parser.add_argument('--multi_evaothers', default=False, action='store_true', help='') 
    parser.add_argument('--A3original', default=False, action='store_true', help='')            
    parser.add_argument('--A3scale', default=False, action='store_true', help='')      
    parser.add_argument('--A3scaleout', type=int, default=1, help="out of nine point")    
    parser.add_argument('--volume4d_group', type=int, default=1, help="")
    parser.add_argument('--A3pos_enconding', default=False, action='store_true', help='')    
    parser.add_argument('--A3require_grad', default=True, action='store_true', help='')    
    parser.add_argument('--A3pos_dim', type=int, default=12, help="width of the correlation pyramid")
    parser.add_argument('--A3mlphidden_list', type=int, nargs='+', default=[64, 32], help="size of the random image crops used during training.")
    parser.add_argument('--A3volume4d_group', type=int, default=0, help="")    
    
    parser.add_argument('--unfold_similarity', default=None, help='')    
    parser.add_argument('--lsp_width', type=int, default=3)
    parser.add_argument('--lsp_height', type=int, default=3)
    parser.add_argument('--lsp_dilation', type=list, default=[1, 2, 4, 8])   
    
    
    parser.add_argument('--mask_invalid', default=False, action='store_true', help='enable mask invalid correlations')
          
    parser.add_argument('--agg_before', default=False, action='store_true', help='')   
    parser.add_argument('--agg_align', default=False, action='store_true', help='')   
    parser.add_argument('--agg_alignv2', default=False, action='store_true', help='')       
    parser.add_argument('--agg_alignv3', default=False, action='store_true', help='')   
    parser.add_argument('--agg_type', default='IGEV', help='') 
    
    parser.add_argument('--quater_nearest', default=None, help='')     
    
    args = parser.parse_args()
    savepath_metrics_all=os.path.join(args.savepath, args.model)
    args.savepath=os.path.join(args.savepath, args.model,args.dataset)
    if os.path.exists(args.savepath):
        paths = os.listdir( args.savepath ) 
        for path in paths:
            if 'events.out.' in path or 'result' in path:
                os.remove(os.path.join(args.savepath,path))
    logger = SummaryWriter(args.savepath)
    os.makedirs(os.path.join(args.savepath,"output"), exist_ok=True)
    if args.model:  
        model = torch.nn.DataParallel(__models__[args.model](args), device_ids=[0])
        if args.restore_ckpt:
            assert args.restore_ckpt.endswith(".pth")
            checkpoint = torch.load(args.restore_ckpt)
            model.load_state_dict(checkpoint, strict=True)
        model.cuda()
        
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(args,logger,model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti15' or args.dataset == 'kitti12':
        validate_kitti(args,logger,model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(args,logger,model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        validate_things(args,logger,model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
    avg_test_scalars = avg_test_scalars.mean()
    avg_test_scalars_occ = avg_test_scalars_occ.mean()
    avg_test_scalars_nonocc = avg_test_scalars_nonocc.mean()  

    print("### {} checkpoint \'{}\'    ".format(args.dataset,args.restore_ckpt)) ; 
    print(" All  {}     ".format(avg_test_scalars))  
    print(" Occ  {}     ".format(avg_test_scalars_occ))
    print(" Noc  {}     ".format(avg_test_scalars_nonocc))   
    if args.record:
        with open(args.savepath+'/result.txt','a',encoding='utf-8') as f:
            f.write(" All {}\n".format(avg_test_scalars))
            f.write(" Occ {}\n".format(avg_test_scalars_occ))            
            f.write(" Noc {}\n".format(avg_test_scalars_nonocc))                   
            
        with open(os.path.join(savepath_metrics_all,'result.txt'),'a',encoding='utf-8') as f: 
            f.write("### {} checkpoint `{}`    \nResults:    \n".format(args.dataset,args.restore_ckpt)) ;        
            f.write("` All  {}`     \n".format(avg_test_scalars))       
            f.write("` Occ  {}`     \n".format(avg_test_scalars_occ))           
            f.write("` Noc  {}`     \n".format(avg_test_scalars_nonocc))   