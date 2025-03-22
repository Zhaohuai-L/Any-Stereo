
from __future__ import print_function, division
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from models.corePrune_RAFT.prune_raft_stereo import continuous_RaftStereo
from evaluation_validate import *
import models.corePrune_RAFT.stereo_datasets as datasets
import torch.nn.functional as F
from metrics_utils.experiment import save_scalars,tensor2float,adjust_learning_rate


try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def sequence_loss(disp_preds, disp_gt, valid, loss_gamma=0.9, max_disp=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    valid = (valid >= 0.5) & (disp_gt < max_disp)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item()
    }

    return disp_loss, metrics

def sequence_loss_multiscale(disp_preds, disp_gt, valid, loss_gamma=0.9, max_disp=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    valid = ((valid >= 0.5) & (disp_gt < max_disp))
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item()
    }

    return disp_loss, metrics

def sequence_loss_multiscale_superinit(init_disp_preds,low_dispgt,disp_preds, disp_gt, valid, loss_gamma=0.9, max_disp=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    valid = ((valid >= 0.5) & (disp_gt < max_disp))
    valid_low=low_dispgt < (max_disp/4.0)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()
    disp_loss +=1.0* F.smooth_l1_loss(init_disp_preds[valid_low.bool()], low_dispgt[valid_low.bool()], size_average=True)
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item()
    }
    return disp_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    if args.lr_fixed:
        scheduler=None
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=os.path.join(args.savepath,'runs'))   
        self.sum_fre=args.sum_fre
    def _print_training_status(self):
        if not args.lr_fixed:
            training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        else:
            training_str = "[{:6d}] ".format(self.total_steps+1)
        metrics_str = " "
        for k in sorted(self.running_loss.keys()):
            metrics_str=metrics_str+"{}:{:.4f} ".format(k,self.running_loss[k]/self.sum_fre)
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(args.savepath,'runs'))

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/self.sum_fre, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.sum_fre == self.sum_fre-1: 
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results,key="Test"):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(args.savepath,'runs'))
        save_scalars(self.writer, key, tensor2float(results),  self.total_steps)
    def close(self):
        self.writer.close()


def train(args):

    if 'sceneflow' in args.train_datasets[0]:
        max_disp_fro_valid=512
    elif 'middlebury' in args.train_datasets[0]:
        max_disp_fro_valid=1000
        print("haha")
    else:
        max_disp_fro_valid=512
    model = nn.DataParallel(PrunedRaftStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")
    model.cuda()
    model.train()
    model.module.freeze_bn()

    validation_frequency = args.valid_fre  

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            if args.multi_training or args.multi_input_training:
                image1, image2,hr_coord, hr_dispgt,arry_scale = [x.cuda() for x in data_blob]
                assert model.training                
                disp_preds = model(image1, image2, iters=args.train_iters,hr_coord=hr_coord,scale=arry_scale)
                assert model.training
                loss, metrics = sequence_loss_multiscale(disp_preds, hr_dispgt, (hr_dispgt < max_disp_fro_valid)&(hr_dispgt>0.),max_disp=args.max_disp)
            else:
                image1, image2, disp_gt = [x.cuda() for x in data_blob]
                assert model.training                   
                init_disp_preds,disp_preds = model(image1, image2, iters=args.train_iters)
                assert model.training                   
                loss, metrics = sequence_loss(disp_preds, disp_gt, (disp_gt < 512)&(disp_gt>0),max_disp=args.max_disp)
                
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            if not args.lr_fixed:
                scheduler.step()
            scaler.update()
            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(os.path.join(args.savepath,'%d_%s.pth'% (total_steps + 1, args.name)))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)
                if 'kitti' in args.train_datasets[0]:                    
                    results_all,results_occ,results_noc = validate_kitti(args,logger=None, model=model.module, iters=args.valid_iters, mode='kitti15')
                    results_noc=results_noc.mean()
                    print("Kitti15 {}".format(results_noc))
                    logger.write_dict(results_noc,key='kitti15')  
                    
                    results_all,results_occ,results_noc = validate_kitti(args,logger=None, model=model.module, iters=args.valid_iters, mode='kitti12')
                    results_noc=results_noc.mean()
                    logger.write_dict(results_noc,key='kitti12')  
                    print("Kitti12 {}".format(results_noc))
                    model.train()
                    model.module.freeze_bn()    
                if 'middlebury' in args.train_datasets[0]:
                    if args.without_mutli_scale:
                        results_all,results_occ,results_noc = validate_middlebury(args,logger=None, model=model.module, iters=args.valid_iters,split='F')
                        results_noc=results_noc.mean()
                        print("middlebury {}".format(results_noc))
                        logger.write_dict(results_noc,key='middlebury_F')                               
                    else:         
                        assert args.scale_min==args.scale_max
                        if args.scale_min<=2.1:
                            results_all,results_occ,results_noc = validate_middlebury(args,logger=None, model=model.module, iters=args.valid_iters,split='H_F')
                            results_noc=results_noc.mean()
                            print("middlebury {}".format(results_noc))
                            logger.write_dict(results_noc,key='middlebury_H_F')                       
                        elif args.scale_min<=4.1:
                            results_all,results_occ,results_noc = validate_middlebury(args,logger=None, model=model.module, iters=args.valid_iters,split='Q_F')
                            results_noc=results_noc.mean()
                            print("middlebury {}".format(results_noc))
                            logger.write_dict(results_noc,key='middlebury_Q_F')   
                        else :
                            assert False     
                    model.train()
                    model.module.freeze_bn()                                   
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path(os.path.join(args.savepath,'%d_epoch_%s.pth.gz' % (total_steps + 1, args.name)))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = os.path.join(args.savepath,'%s.pth' % args.name)
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='prune-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--savepath', default=None, help='save path')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--lr_fixed', default=False, action='store_true', help='')     #
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.") 
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--sum_fre', type=int, default=100, help='number of flow-field updates during validation forward pass')
    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--valid_fre', type=int, default=10000, help='number of flow-field updates during validation forward pass')    
    parser.add_argument('--scale_test',type=int, default=1, help="the scale number for test, must lager than 1") 
    parser.add_argument('--max_disp', type=int, default=700, help="max disp of geometry encoding volume") 
    parser.add_argument('--max_enable', help='enable the max constraint', action='store_true', default=False) 
    parser.add_argument('--record', help='record on text', action='store_true', default=False) 
    parser.add_argument('--ShowImage', help='Show IMage', action='store_true', default=False)
    parser.add_argument('--output', help='Ouput the image as colored png', action='store_true',default=False)        
    parser.add_argument('--model', type=str, default='continuous_RAFTStereo')               
    
    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid") 
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions") 
    
    #ISU
    parser.add_argument('--unfold_similarity', default="with_v2ISU", help='') 
    parser.add_argument('--lsp_width', type=int, default=3)
    parser.add_argument('--lsp_height', type=int, default=3)
    parser.add_argument('--lsp_dilation', type=list, default=[1, 2, 4, 8]) 
    
    
    #Implict upsampling
    parser.add_argument('--local_ensemble', default=False, action='store_true', help='')    
    parser.add_argument('--decode_cell', default=False, action='store_true', help='')   
    parser.add_argument('--unfold', default=False, action='store_true', help='')      
    parser.add_argument('--Raw_Mask_dim', type=int, default=32, help="")
        
    #pos encoding 
    parser.add_argument('--pos_enconding_new', default=False, action='store_true', help='')
    parser.add_argument('--pos_enconding', default=False, action='store_true', help='')
    parser.add_argument('--require_grad', default=False, action='store_true', help='')
    parser.add_argument('--pos_dim', type=int, default=0, help="")
    parser.add_argument('--mlphidden_list', type=int, nargs='+', default=[128,64,64], help="")
    #Multi-Scale Training 
    parser.add_argument('--multi_training', default=False, action='store_true', help='')
    parser.add_argument('--inp_size',type=int, nargs='+', default=[160,320], help="")
    parser.add_argument('--scale_min',type=float, default=1, help="")
    parser.add_argument('--scale_max',type=float, default=2.95, help="")  
    parser.add_argument('--without_mutli_scale', default=False, action='store_true', help='') 
    
    
    #Multi-Input Training 
    parser.add_argument('--multi_input_training', default=False, action='store_true', help='')  
           
    #Disparity Norm 
    parser.add_argument('--disparity_norm', default=False, action='store_true', help='')
    parser.add_argument('--multi_evaothers', default=False, action='store_true', help='')                    
                     
    #Quarter Nearest 
    parser.add_argument('--quater_nearest', default=None, help='') 
                 
    #CFA
    parser.add_argument('--agg_type', default='type5', help='')
             
    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.savepath).mkdir(exist_ok=True, parents=True)     

    train(args)