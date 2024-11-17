import torch
import torch.nn.functional as F
from metrics_utils.experiment import make_nograd_func
from torch.autograd import Variable
from torch import Tensor


# Update D1 from >3px to >=3px & >5%
# matlab code:
# E = abs(D_gt - D_est);
# n_err = length(find(D_gt > 0 & E > tau(1) & E. / abs(D_gt) > tau(2)));
# n_total = length(find(D_gt > 0));
# d_err = n_err / n_total;

def check_shape_for_metric_computation(*vars):
    assert isinstance(vars, tuple)
    for var in vars:
        assert len(var.size()) == 3
        assert var.size() == vars[0].size()

# a wrapper to compute metrics for each image individually
def compute_metric_for_each_image(metric_func):
    def wrapper(D_ests, D_gts, masks, *nargs):
        check_shape_for_metric_computation(D_ests, D_gts, masks)
        bn = D_gts.shape[0]  # batch size
        results = []  # a list to store results for each image
        # compute result one by one
        for idx in range(bn):
            # if tensor, then pick idx, else pass the same value
            cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
            #     print("masks[idx].float().mean() too small, skip")
            # else:
            #     ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
            #     results.append(ret)
            ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
            results.append(ret)
        if len(results) == 0:
            print("masks[idx].float().mean() too small for all images in this batch, return 0")
            return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
        else:
            return torch.stack(results).mean()
    return wrapper

def compute_metric_for_each_image_filter_NULL(metric_func):
    def wrapper(D_ests, D_gts, masks, *nargs):
        check_shape_for_metric_computation(D_ests, D_gts, masks)
        bn = D_gts.shape[0]  # batch size
        results = []  # a list to store results for each image
        # compute result one by one
        for idx in range(bn):
            # if tensor, then pick idx, else pass the same value
            cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
            if masks[idx].float().mean() / (D_gts[idx] > 0).float().mean() < 0.01:
                print("masks[idx].float().mean() too small, skip")
            else:
                ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
                results.append(ret)
            # results.append(ret)
        if len(results) == 0:
            print("masks[idx].float().mean() too small for all images in this batch, return 0")
            return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
        else:
            return torch.stack(results).mean()
    return wrapper

@make_nograd_func
@compute_metric_for_each_image
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metric_for_each_image
def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metric_for_each_image
def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    loss=F.l1_loss(D_est, D_gt, size_average=True)
    # print("Epe loss is {}".format(loss))
    return loss



@make_nograd_func
@compute_metric_for_each_image_filter_NULL
def D1_metric_filter(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metric_for_each_image_filter_NULL
def Thres_metric_filter(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metric_for_each_image_filter_NULL
def EPE_metric_filter(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    loss=F.l1_loss(D_est, D_gt, size_average=True)
    # print("Epe loss is {}".format(loss))
    return loss




@make_nograd_func
@compute_metric_for_each_image
def D1_metric_mask(D_est, D_gt, mask, mask_img):
    # D_est, D_gt = D_est[(mask&mask_img)], D_gt[(mask&mask_img)]
    D_est, D_gt = D_est[mask_img], D_gt[mask_img]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metric_for_each_image
def Thres_metric_mask(D_est, D_gt, mask, thres, mask_img):
    assert isinstance(thres, (int, float))
    # D_est, D_gt = D_est[(mask&mask_img)], D_gt[(mask&mask_img)]
    D_est, D_gt = D_est[mask_img], D_gt[mask_img]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metric_for_each_image
def EPE_metric_mask(D_est, D_gt, mask, mask_img):
    # print((mask&mask_img).size(), D_est.size(), mask, mask_img)
    # D_est, D_gt = D_est[(mask&mask_img)], D_gt[(mask&mask_img)]
    D_est, D_gt = D_est[mask_img], D_gt[mask_img]
    return F.l1_loss(D_est, D_gt, size_average=True)


@make_nograd_func
def compute_iou(pred: Tensor, occ_mask: Tensor, invalid_mask: Tensor):
    """
    compute IOU on occlusion

    :param pred: occlusion prediction [N,H,W]
    :param occ_mask: ground truth occlusion mask [N,H,W]
    :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
    """
    # threshold
    pred_mask = pred 
    # iou for occluded region
    inter_occ = torch.logical_and(pred_mask, occ_mask).sum()
    union_occ = torch.logical_or(torch.logical_and(pred_mask, ~invalid_mask), occ_mask).sum()

    # iou for non-occluded region
    inter_noc = torch.logical_and(~pred_mask, ~invalid_mask).sum()
    union_noc = torch.logical_or(torch.logical_and(~pred_mask, occ_mask), ~invalid_mask).sum()

    # aggregate
    iou= (inter_occ + inter_noc).float() / (union_occ + union_noc)

    return iou