from __future__ import print_function, division

from torch.autograd import Variable

try:

    from itertools import  ifilterfalse

except ImportError: # py3k

    from itertools import  filterfalse as ifilterfalse
import os
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import scipy.io as sio
from skimage import feature
from collections import Counter
from IPython.core import debugger
debug = debugger.Pdb().set_trace
from torch.autograd import Variable, Function
def normalize(data):
    batch = data.size(0)
    for i in range(batch):
        data[i] =  (data[i] - data[i].min()) / (data[i].max() - data[i].min())
    return data


def colortojet(img):
    cmap=plt.get_cmap('jet')
    img = img.numpy()
    img_jet = cmap(img)[:,:,:,0:3]
    img_out = torch.from_numpy(img_jet).transpose(0,3).squeeze(-1).float()
    return img_out


def AGC_mats(data, dt=1.0*1.0e-5, parameters=0.05, options=1):
    num = data.shape[0]
    out = data.copy()
    for i in range(num):
        out[i,0] = AGC_mat(data[i,0], dt, parameters, options)
        print("num: "+str(i+1))
    return out


def AGC_mat(data, dt=1.0*1.0e-5, parameters=0.05, options=1):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    mat_data = matlab.double(data.tolist())
    agc_data = eng.AGCgain(mat_data, dt, parameters, options)
    agc_data = np.array(agc_data)
    return agc_data


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]
    return lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.minimum = 1000000
        self.maximum = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.minimum = val if self.minimum > val else self.minimum
        self.maximum = val if self.maximum < val else self.maximum
        

def to_tensorboard(lock, writer, img, iter_num, tag='Obs&Pre', nrow=8, norm=False):
    lock.acquire()
    show_img = vutils.make_grid(img, nrow=nrow, padding=2, normalize=norm, scale_each=norm, pad_value=0)
    writer.add_image(tag, show_img, iter_num)           
    lock.release()

    
def visualize_img(input, target, prob):
    img = torch.zeros_like(input, device='cpu', requires_grad=False)
    img.copy_(input)
    img = img.repeat(1,3,1,1)
    img = F.interpolate(img, size=200)
    batchsize = input.size(0)
    for i in range(batchsize):
        img_pil = transforms.functional.to_pil_image(norm(img[i]))
        img_draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype('Arial.ttf', 20)
        img_draw.text((0, 0),str(target[i].item())+'/'+str("%.4f" % prob[i].item()),(255,0,0),font=font)
        img[i] = transforms.functional.to_tensor(img_pil)
    return img    
    

def save_img(lock, img, path, nrow=8, norm=False):
    lock.acquire()
    batch = img.size(0)
    for i in range(batch):
        tpath = path[i].replace('valid','valid_predict').replace('test','test_predict').replace('train','train_predict').replace('mat','png')
        File_Path = tpath[0:tpath.rfind('/')]
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)
        vutils.save_image(img[i], tpath, nrow=nrow, padding=0, normalize=norm, scale_each=norm)
    lock.release()


def save_mat(data, path, postfix):
    tpath = path[0].replace('valid','valid_predict_mat').replace('test','test_predict_mat').replace('train','train_predict_mat').replace('ceshi','ceshi_predict_mat').replace('yuce','yuce_predict_mat')
    File_Path = tpath[0:tpath.rfind('/')]
    if not os.path.exists(File_Path):
        os.makedirs(File_Path)
    filename = tpath[0:tpath.rfind('.')] + '_' + postfix + '.mat'
    sio.savemat(filename, {postfix: data})


def save_feature(lock, img, feature, path):
    lock.acquire()
    batch = img.size(0)
    for i in range(batch):
        img_path = path[i].replace('valid','valid_predict').replace('test','test_predict').replace('train','train_predict').replace('mat','png')
        feat_path = path[i].replace('valid','valid_predict').replace('test','test_predict').replace('train','train_predict').replace('.mat','_feat.png')
        File_Path = img_path[0:img_path.rfind('/')]
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)
        vutils.save_image(img[i], img_path, nrow=1, padding=0, normalize=False, scale_each=False)
        vutils.save_image(feature[i].unsqueeze(1), feat_path, nrow=20, padding=0, normalize=True, scale_each=True)
    lock.release()


def edge_detect(x):   
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    h_tv = torch.pow((x[:,:,1:,:-1]-x[:,:,:h_x-1,:-1]),2).sum(1, True)
    w_tv = torch.pow((x[:,:,:-1,1:]-x[:,:,:-1,:w_x-1]),2).sum(1, True)
    return F.pad(h_tv + w_tv, (0, 1, 0, 1), 'replicate')    
 

def canny(im):
    edge = feature.canny(im.squeeze().cpu().numpy()).astype(np.float16)
    return torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).cuda()

    
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    

class SegLoss(nn.Module):
    def __init__(self, ignore_label=-100):
        super(SegLoss, self).__init__()
        self.obj = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
    def forward(self, pred, label):        
        loss = self.obj(pred, label)
        return loss    
    

class EdgeLoss(nn.Module):
    def __init__(self, mode='normal'):
        super(EdgeLoss, self).__init__()
        self.mode = mode
    def forward(self, pred, target):
        #pred_edge = torch.tanh(edge_detect(pred))
        #target = nn.functional.interpolate(target, pred_edge.size()[2:3])
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1) * target
        if self.mode == 'normal':
            loss = F.binary_cross_entropy(pred, target, weights)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target, weights)
        #loss = nn.functional.mse_loss(pred_edge, target)
        return loss


class EdgeLossVanila(nn.Module):
    def __init__(self, mode='normal'):
        super(EdgeLossVanila, self).__init__()
        self.mode = mode
    def forward(self, pred, target):
        if self.mode == 'normal':
            loss = F.binary_cross_entropy(pred, target)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss


class SoftEdgeLoss(nn.Module):
    def __init__(self, mode='normal'):
        super(SoftEdgeLoss, self).__init__()
        self.mode = mode
    def forward(self, pred, target):
        kernel = torch.from_numpy(gkern(kernlen=5)).float().unsqueeze(0).unsqueeze(0).cuda()
        pred = nn.functional.pad(pred, (2,2,2,2), 'replicate')
        pred = F.conv2d(pred, kernel)
        # target = (target-target.min())/(target.max()-target.min())
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1) * target
        if self.mode == 'normal':
            loss = F.binary_cross_entropy(pred, target, weights)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target, weights)
        return loss


class SoftIOU(nn.Module):
    def __init__(self):
        super(SoftIOU, self).__init__()
    def forward(self, pred, target):
        kernel = torch.from_numpy(gkern(kernlen=7)).float().unsqueeze(0).unsqueeze(0).cuda()
        pred_soft = nn.functional.pad(pred, (3,3,3,3), 'replicate')
        pred_soft = F.conv2d(pred_soft, kernel)
        pred_soft = (pred_soft - pred_soft.min()) / (pred_soft.max() - pred_soft.min())
        if torch.sum(pred) <= 0:
            #torch.sum(torch.isnan(TP))>0:
            #debug()
            Fmeasure = torch.zeros(1)  
        else:
            TP = torch.sum(pred_soft[target == 1])
            Precise = TP / torch.sum(pred)
            Recall = TP / torch.sum(target)
            Fmeasure = 2 * (Precise * Recall) / (Precise + Recall)      
        return Fmeasure #, Precise, Recall


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def TransformedDomainRecursiveFilter(Img, D, sigma):
    sqr2 = 1.4142135623730951
    a = torch.exp(torch.Tensor([-sqr2/sigma])).cuda()
    V = torch.pow(a,D)
    b,c,h,w = Img.shape
    for i in range(1,w,1):
        Img[:,:,:,i] = Img[:,:,:,i] + V[:,:,:,i] * (Img[:,:,:,i-1] - Img[:,:,:,i])
    for i in range(w-2,-1,-1):
        Img[:,:,:,i] = Img[:,:,:,i] + V[:,:,:,i+1] * (Img[:,:,:,i+1] - Img[:,:,:,i])
    return Img


class Domain_Transform(nn.Module):
    def __init__(self, sigma_s = 60, sigma_r = 0.4, iter_num=2):
        super(Domain_Transform, self).__init__()
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        #self.sigma_s = nn.Parameter(torch.Tensor(1))
        #self.sigma_r = nn.Parameter(torch.Tensor(1))
        #self.sigma_s.data.fill_(sigma_s)
        #self.sigma_r.data.fill_(sigma_r)
        self.iter_num = iter_num
    def forward(self, img, edge):      ###########forward?
        dHdx = (1 + self.sigma_s/self.sigma_r * edge)
        dVdy = dHdx.transpose(2,3)
        sqr3 = 1.7320508075688772
        for i in range(self.iter_num):
            sigma_i = self.sigma_s*sqr3*torch.pow(torch.Tensor([2]),self.iter_num-(i+1))/torch.sqrt(torch.pow(torch.Tensor([4]),self.iter_num)-1);
            img = TransformedDomainRecursiveFilter(img, dHdx, sigma_i);
            img = img.transpose(2,3)
            img = TransformedDomainRecursiveFilter(img, dVdy, sigma_i);
            img = img.transpose(2,3)
        return img


def gaussian(window_size, sigma):
    gauss = torch.exp(torch.Tensor([-(x - window_size//2)**2/float(2*sigma**2) for x in range(window_size)]))
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window.to(img1.device), window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_

def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
 
    sum_k_t_k = get_pixel_area(eval_segm)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_

'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)



def lovasz_grad(gt_sorted):

    """

    Computes gradient of the Lovasz extension w.r.t sorted errors

    See Alg. 1 in paper

    """

    p = len(gt_sorted)

    gts = gt_sorted.sum()

    intersection = gts - gt_sorted.float().cumsum(0)

    union = gts + (1 - gt_sorted).float().cumsum(0)

    jaccard = 1. - intersection / union

    if p > 1: # cover 1-pixel case

        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard





def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):

    """

    IoU for foreground class

    binary: 1 foreground, 0 background

    """

    if not per_image:

        preds, labels = (preds,), (labels,)

    ious = []

    for pred, label in zip(preds, labels):

        intersection = ((label == 1) & (pred == 1)).sum()

        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()

        if not union:

            iou = EMPTY

        else:

            iou = float(intersection) / float(union)

        ious.append(iou)

    iou = mean(ious)    # mean accross images if per_image

    return 100 * iou





def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):

    """

    Array of IoU for each (non ignored) class

    """

    if not per_image:

        preds, labels = (preds,), (labels,)

    ious = []

    for pred, label in zip(preds, labels):

        iou = []    

        for i in range(C):

            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)

                intersection = ((label == i) & (pred == i)).sum()

                union = ((label == i) | ((pred == i) & (label != ignore))).sum()

                if not union:

                    iou.append(EMPTY)

                else:

                    iou.append(float(intersection) / float(union))

        ious.append(iou)

    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image

    return 100 * np.array(ious)





# --------------------------- BINARY LOSSES ---------------------------





def lovasz_hinge(logits, labels, per_image=True, ignore=None):

    """

    Binary Lovasz hinge loss

      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)

      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)

      per_image: compute the loss per image instead of per batch

      ignore: void class id

    """

    if per_image:

        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))

                          for log, lab in zip(logits, labels))

    else:

        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))

    return loss





def lovasz_hinge_flat(logits, labels):

    """

    Binary Lovasz hinge loss

      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)

      labels: [P] Tensor, binary ground truth labels (0 or 1)

      ignore: label to ignore

    """

    if len(labels) == 0:

        # only void pixels, the gradients should be 0

        return logits.sum() * 0.

    signs = 2. * labels.float() - 1.

    errors = (1. - logits * Variable(signs))

    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)

    perm = perm.data

    gt_sorted = labels[perm]

    grad = lovasz_grad(gt_sorted)

    loss = torch.dot(F.relu(errors_sorted), Variable(grad))

    return loss





def flatten_binary_scores(scores, labels, ignore=None):

    """

    Flattens predictions in the batch (binary case)

    Remove labels equal to 'ignore'

    """

    scores = scores.view(-1)

    labels = labels.view(-1)

    if ignore is None:

        return scores, labels

    valid = (labels != ignore)

    vscores = scores[valid]

    vlabels = labels[valid]

    return vscores, vlabels





class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):

         super(StableBCELoss, self).__init__()

    def forward(self, input, target):

         neg_abs = - input.abs()

         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()

         return loss.mean()





def binary_xloss(logits, labels, ignore=None):

    """

    Binary Cross entropy loss

      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)

      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)

      ignore: void class id

    """

    logits, labels = flatten_binary_scores(logits, labels, ignore)

    loss = StableBCELoss()(logits, Variable(labels.float()))

    return loss





# --------------------------- MULTICLASS LOSSES ---------------------------
def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   º∆À„dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss






def _smooth_l1_loss(x, t):
        in_weight=0.08 
        diff = (in_weight * (x - t)).abs()
        # abs_diff = diff.abs()
        flag = (diff < 1).float()
        y = (flag * 0.5 * (diff ** 2) +
             (1 - flag) * (diff - 0.5))
        return y.sum()


# huber À ß
def huber(true, pred, delta):
    loss = torch.where(torch.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*torch.abs(true - pred) - 0.5*(delta**2))

    return mean(loss)







def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):

    """

    Multi-class Lovasz-Softmax loss

      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).

              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].

      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)

      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.

      per_image: compute the loss per image instead of per batch

      ignore: void class labels

    """

    if per_image:

        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)

                          for prob, lab in zip(probas, labels))

    else:

        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)

    return loss





def lovasz_softmax_flat(probas, labels, classes='present'):

    """

    Multi-class Lovasz-Softmax loss

      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)

      labels: [P] Tensor, ground truth labels (between 0 and C - 1)

      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.

    """

    if probas.numel() == 0:

        # only void pixels, the gradients should be 0

        return probas * 0.

    C = probas.size(1)

    losses = []

    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes

    for c in class_to_sum:

        fg = (labels == c).float() # foreground for class c

        if (classes is 'present' and fg.sum() == 0):

            continue

        if C == 1:

            if len(classes) > 1:

                raise ValueError('Sigmoid output possible only with 1 class')

            class_pred = probas[:, 0]

        else:

            class_pred = probas[:, c]

        errors = (Variable(fg) - class_pred).abs()

        errors_sorted, perm = torch.sort(errors, 0, descending=True)

        perm = perm.data

        fg_sorted = fg[perm]

        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))

    return mean(losses)





def flatten_probas(probas, labels, ignore=None):


    if probas.dim() == 3:

        # assumes output of a sigmoid layer

        B, H, W = probas.size()

        probas = probas.view(B, 1, H, W)

    B, C, H, W = probas.size()

    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C

    labels = labels.view(-1)

    if ignore is None:

        return probas, labels

    valid = (labels != ignore)

    vprobas = probas[valid.nonzero().squeeze()]

    vlabels = labels[valid]

    return vprobas, vlabels



def xloss(logits, labels, ignore=None):

    """

    Cross entropy loss

    """

    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

def isnan(x):

    return x != x

    

    

def mean(l, ignore_nan=False, empty=0):

    """

    nanmean compatible with generators.

    """

    l = iter(l)

    if ignore_nan:

        l = ifilterfalse(isnan, l)

    try:

        n = 1

        acc = next(l)

    except StopIteration:

        if empty == 'raise':

            raise ValueError('Empty mean')

        return empty

    for n, v in enumerate(l, 2):

        acc += v

    if n == 1:

        return acc

    return acc / n