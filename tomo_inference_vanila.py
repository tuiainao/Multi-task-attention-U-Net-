import os
import sys
import time
import random
import socket
import threading
import numpy as np
from datetime import datetime
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchnet as tnt
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
from tensorboardX import SummaryWriter

import basic_function2 as func
import tomo_dataset2 as dataset

from IPython.core import debugger
debug = debugger.Pdb().set_trace

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

args_seed = 666
args_works = 6
args_valid = '/media/lm/shuju/test/'
args_gpu = True
args_batch = 1
mode = 'tomo'

if mode == 'tomo':
    import tomo_model as model
else:
    import inversenet_model2 as model

random.seed(args_seed)
torch.manual_seed(args_seed)
torch.cuda.manual_seed_all(args_seed)
#cudnn.deterministic = True
cudnn.benchmark = True
device = torch.device("cuda")

valid_dataset = dataset.DatasetFolder(args_valid, flip=False, norm=True)
valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args_batch, shuffle=False,
        num_workers=args_works, pin_memory=True, drop_last=False)

if mode == 'tomo':
    NetWorks = model.TomoNet().to(device)
else:
    NetWorks = model.InverseNet().to(device)

NetWorks = nn.DataParallel(NetWorks, device_ids=range(torch.cuda.device_count()))
checkpoint_file = torch.load('/media/lm/lmcx/lm106/runs/Oct06_20-34-58zjq5a97b294d918/minloss_checkpoint.pth.tar')
NetWorks.load_state_dict(checkpoint_file['state_dict'])

ssim = func.MSSSIM()
#Edge = func.SoftIOU()
#SSIM = func.SSIM()
#MSSIM = func.MSSSIM()
class_total = list(0. for i in range(3))
class_total =torch.Tensor(class_total)
def valid(valid_loader, model):
    losses1 = func.AverageMeter()
    losses2 = func.AverageMeter()
    losses3 = func.AverageMeter()
    losses4 = func.AverageMeter()
    losses = func.AverageMeter()
    #losses_ssim = func.AverageMeter() 
    ##losses_mssim = func.AverageMeter()
    
    # switch to train mode
    model.eval()
    with torch.no_grad():
        accuracy1 = 0
        accuracy2 = 0
        accuracy3 = 0
        precision = list(0. for i in range(3))
        precision=torch.Tensor(precision)
        recall = list(0. for i in range(3)) 
        recall=torch.Tensor(recall)
        for step, (observe, geology,jiedian, observe_path, geology_path,jiedian_path) in enumerate(valid_loader):
            observe, geology,jiedian = observe.to(device), geology.to(device),jiedian.to(device)
            if mode != 'tomo':
                observe = observe.transpose(2,3)
            #print(observe_path)
            #geo_edge = func.edge_detect(geology)!=0
            #geo_edge = func.canny(geology)!=0
            #geo_edge = geo_edge.float()

            # compute output
            if mode == 'tomo':
                predict,pre= model(observe)
            else:
                predict,pre= model(observe)

            #jiedian = nn.Upsample(size=[200, 200], mode='bilinear', align_corners=True)(jiedian)
            #geology = nn.Upsample(size=[200, 200], mode='bilinear', align_corners=True)(geology)
            #predict = predict[:,:,10:-10,10:-10]
            #jiedian = jiedian[:,:,:1:,:1]
            #jiedian=jiedian.view(jiedian.size(0),-1)
            predict1 = F.softmax(predict,dim = 1)
            
            loss1 = func.lovasz_softmax(predict1,geology)
            loss2 = func.xloss(predict,geology.squeeze(0).long())
            #loss3 = func.huber(jiedian,predict22,1)
            #loss2 = func.Dice_loss(geology.squeeze(0).long(),predict, beta=1, smooth = 1e-5)
            #debug()
            loss3 = 1 - ssim(pre, jiedian)
            
            L1_loss = nn.L1Loss()
            loss4  =  L1_loss(pre, jiedian)
            
            loss = loss1 + loss2 +loss3+loss4*10
            
            nb_classes = 3
            confusion_matrix = torch.zeros(nb_classes, nb_classes)
            _, preds = torch.max(predict, 1)
            for t, p in zip(geology.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            precision1 = confusion_matrix.diag()/(confusion_matrix.sum(0)+1e-10)
            
            precision += precision1
            recall1 = confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-10)
            recall += recall1
            predict3 = predict.data.max(1, keepdim=True)[1].byte()
            predict333=pre*8+1
            #jiedian2=(jiedian+1)*10+10
            func.save_mat(predict3.squeeze().cpu().numpy(), observe_path, 'predict')
            func.save_mat(predict333.squeeze().cpu().numpy(), observe_path, 'fanyan')
            #func.save_mat(jiedian2.cpu().numpy(), observe_path, 'jiedian')
            acc1 = func.mean_accuracy(predict3.squeeze().cpu().numpy(),geology.squeeze().cpu().numpy())
            acc2 = func.mean_IU(predict3.squeeze().cpu().numpy(),geology.squeeze().cpu().numpy())
            acc3 = func.pixel_accuracy(predict3.squeeze().cpu().numpy(),geology.squeeze().cpu().numpy())
            #func.save_mat(feature.squeeze().cpu().numpy(), observe_path, 'features') cichuyouxiugai!!2019.04.04 11:17 jiyintao!!
            accuracy1 +=acc1
            accuracy2 +=acc2
            accuracy3 +=acc3
            class_total[geology.squeeze(1).long()] +=1
            #debug()
            #loss_l1 = L1(predict, geology)
            #loss_l2 = L2(predict, geology)
            #loss_ssim = SSIM(predict, geology)
            #loss_mssim = MSSIM(predict, geology)

            # measure accuracy and record loss
            losses1.update(loss1.item(), observe.size(0)) 
            losses2.update(loss2.item(), observe.size(0))
            losses3.update(loss3.item(), observe.size(0))
            losses4.update(loss4.item(), observe.size(0))
            losses.update(loss.item(), observe.size(0))
            #losses_mssim.update(loss_mssim.item(), observe.size(0)) 
                    
    precision = precision / class_total
    recall = recall / class_total
    b = (precision + recall + 1e-10)
    f1 = 2*precision*recall/ b
    print('mean_accuracy : %d %%' % (100 * accuracy1 / len(valid_loader)))
    print('mean_IU: %d %%' % (100 * accuracy2 / len(valid_loader)))
    print('pixel_accuracy : %d %%' % (100 * accuracy3 / len(valid_loader)))
    print('precision : ' ,precision)
    print('recall : ', recall)
    print('bf: ',f1)
    return losses1,losses2,losses3,losses4,losses




losses1,losses2,losses3,losses4,losses = valid(valid_loader, NetWorks)

print('loss_l1 avg: {:.6f}, min: {:.6f}, max: {:.6f}'.format(losses1.avg, losses1.minimum, losses1.maximum)) 
print('loss_l2 avg: {:.6f}, min: {:.6f}, max: {:.6f}'.format(losses2.avg, losses2.minimum, losses2.maximum))
print('loss_l3 avg: {:.6f}, min: {:.6f}, max: {:.6f}'.format(losses3.avg, losses3.minimum, losses3.maximum))
print('loss_l4 avg: {:.6f}, min: {:.6f}, max: {:.6f}'.format(losses4.avg, losses4.minimum, losses4.maximum))
print('loss avg: {:.6f}, min: {:.6f}, max: {:.6f}'.format(losses.avg, losses.minimum, losses.maximum))
#print('loss_mssim avg: {:.6f}, min: {:.6f}, max: {:.6f}'.format(losses_mssim.avg, losses_mssim.minimum, losses_mssim.maximum))
