import os
import sys
import time
import random
import socket
import threading
import numpy as np
from datetime import datetime
import pdb
import torch
import torch.nn as nn
import torchnet as tnt
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
from tensorboardX import SummaryWriter
from collections import Counter
import basic_function2 as func
import tomo_dataset2 as dataset
import tomo_model as model
import torch.nn.functional as F
from IPython.core import debugger
debug = debugger.Pdb().set_trace

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

args_seed = 666
args_lr = 5e-4
args_works = 8
args_epochs = 200          ###200
args_train = '/media/lm/shuju/train/'
args_valid = '/media/lm/shuju/val/'
args_gpu = True
args_freq = 10
args_batch = 16

random.seed(args_seed)
torch.manual_seed(args_seed)
torch.cuda.manual_seed_all(args_seed)
#cudnn.deterministic = True
cudnn.benchmark = True
device = torch.device("cuda")

train_dataset = dataset.DatasetFolder(args_train, flip=True, norm=True)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args_batch, shuffle=True,
        num_workers=args_works, pin_memory=True, drop_last=True)
max_step = args_epochs * len(train_loader)
#debug()
 
valid_dataset = dataset.DatasetFolder(args_valid, flip=False, norm=True)
valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args_batch, shuffle=False,
        num_workers=args_works, pin_memory=True, drop_last=True)


TomoNet = model.TomoNet().to(device)
TomoNet = nn.DataParallel(TomoNet, device_ids=range(torch.cuda.device_count()))
#value = nn.L1Loss()
#edge = func.EdgeLoss('logits')
#edge = func.EdgeLossVanila('logits')
#edge = func.SoftEdgeLoss('logits')
ssim = func.MSSSIM()

optimizer = torch.optim.Adam(TomoNet.parameters(), args_lr)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.1)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.1**(step%15))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.1 * (1.0-float(step)/max_step)**0.9)

#args_momentum = 0.9
#args_weight_decay = 1e-4
#optimizer = torch.optim.SGD(TomoNet.parameters(), args_lr, momentum=args_momentum, weight_decay=args_weight_decay)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.1 * (1.0-float(step)/max_step)**0.9)

date = datetime.now().strftime('%b%d_%H-%M-%S')+'zjq'+socket.gethostname()
save_path = './runs/' + date
writer = SummaryWriter(save_path)



def train(train_loader, model, optimizer, scheduler, epoch, writer):
    lock = threading.Lock()
    threadlist = []
    batch_time = func.AverageMeter() 
    losses = func.AverageMeter()
    losses1 = func.AverageMeter()
    losses2 = func.AverageMeter() 
    losses3 = func.AverageMeter()
    losses4 = func.AverageMeter()    
    epoch_end = time.time()
    
    # switch to train mode
    model.train()
    with torch.enable_grad():
        end = time.time()
        correct = 0
        total1 = 0
        #pdb.set_trace()
       
        for step, (observe, geology,jiedian, observe_path, geology_path,jiedian_path) in enumerate(train_loader):
            observe, geology,jiedian = observe.to(device), geology.to(device),jiedian.to(device)
            #debug()
            current_lr = scheduler.get_lr()[0]
            predict,pre= model(observe)
            
            #jiedian = jiedian[:,:,:1:,:1]
            #jiedian=jiedian.view(jiedian.size(0),-1)
            
            #jiedian = nn.Upsample(size=[200, 200], mode='bilinear', align_corners=True)(jiedian)
            #debug()
            #predict22 = slim.fully_connected(predict22, 4096, scope='predict22')
            #predict22=nn.Linear(1024,640)
            #geology = nn.Upsample(size=[200, 200], mode='bilinear', align_corners=True)(geology)
            
            
            #out1 = out1[:,:,14:84,:]   ##14 114???
            #out2 = out2[:,:,14:84,:]
            #out3 = out3[:,:,14:84,:]
            #predict = predict[:,:,10:-10,10:-10]
            
            predict1 = F.softmax(predict,dim = 1)
            loss1 = func.lovasz_softmax(predict1,geology) 
            #loss2 = func.Dice_loss(geology,predict, beta=1, smooth = 1e-5)
            
            loss2 = func.xloss(predict,geology.squeeze().long())
            
            
            loss3 = 1 - ssim(pre, jiedian)
            
            L1_loss = nn.L1Loss()
            loss4  =  L1_loss(pre, jiedian)
            #loss4 = func._smooth_l1_loss(predict22,jiedian)
            #mmm=predict22-jiedian
            #debug()
            #loss3 = func.huber(jiedian,predict22,1)
            #jiedian1 = torch.abs(jiedian-predict22)
            #debug()
            loss = loss3+loss4*10+loss1+loss2
            # measure accuracy and record loss
            losses1.update(loss1.item(), observe.size(0)) 
            losses2.update(loss2.item(), observe.size(0)) 
            losses3.update(loss3.item(), observe.size(0))
            losses4.update(loss4.item(), observe.size(0))
            losses.update(loss.item(), observe.size(0))    

            # compute gradient and do SGD step         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            predict3 = predict.data.max(1, keepdim=True)[1].byte()
            geology2 = geology.flatten(0)
            predict3 = predict3.to(torch.float32) 
            correct += (predict3 == geology).sum().item()
            total1 += geology2.size(0)
             #measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            writer.add_scalar('loss_train', loss.item(), epoch*len(train_loader)+step)
            writer.add_scalar('lr', current_lr, epoch*len(train_loader)+step)
            
            if step % args_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Lr {lr:.8f}\t'.format(
                       epoch, step, len(train_loader), batch_time=batch_time,
                       loss=losses, lr=current_lr))
                #print('Accuracy : %d %%' % (
    #100 * correct / total1))

            if step % (args_freq * 10) == 0:
                threadlist.append(threading.Thread(target=func.to_tensorboard, args=(lock, writer, torch.cat((jiedian, pre),0).cpu(), epoch*len(train_loader)+step, 'img_train', args_batch)))
                threadlist[-1].start()
        print('main thread finished, waiting for IO threads...')
        for thread in threadlist:
            thread.join()

        epoch_time = time.time() - epoch_end
        print('Train time : {:.2f}min'.format(epoch_time/60))         
        writer.add_scalar('Loss1_Train', losses1.avg, epoch)        
        writer.add_scalar('Loss2_Train', losses2.avg, epoch)
        writer.add_scalar('Loss3_Train', losses3.avg, epoch)
        writer.add_scalar('Loss4_Train', losses4.avg, epoch)                   
        writer.add_scalar('Loss_Train', losses.avg, epoch)

#class_correct = list(0. for i in range(11))  #0.0 00000000
#class_total = list(0. for i in range(11))

def valid(valid_loader, model, epoch, writer):
    lock = threading.Lock()
    threadlist = []
    batch_time = func.AverageMeter()
    losses = func.AverageMeter()
    losses1 = func.AverageMeter()
    losses2 = func.AverageMeter()
    losses3 = func.AverageMeter() 
    losses4 = func.AverageMeter()  
    epoch_end = time.time()
    
    # switch to train mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        correct = 0
        total1 = 0
        for step, (observe, geology,jiedian, observe_path, geology_path,jiedian_path) in enumerate(train_loader):
            observe, geology,jiedian = observe.to(device), geology.to(device),jiedian.to(device)

            #geo_edge = func.edge_detect(geology)!=0
            #geo_edge = geo_edge.float()
            
            # compute output
            predict,pre= model(observe)
            #jiedian = jiedian[:,:,:1:,:1]
            #jiedian=jiedian.view(jiedian.size(0),-1)
            #jiedian = nn.Upsample(size=[200, 200], mode='bilinear', align_corners=True)(jiedian)
            #geology = nn.Upsample(size=[200, 200], mode='bilinear', align_corners=True)(geology)
            #out1 = out1[:,:,14:84,:]   ###14 114?
            #out2 = out2[:,:,14:84,:]
            #out3 = out3[:,:,14:84,:]
            #predict = predict[:,:,10:-10,10:-10]
            predict1 = F.softmax(predict,dim = 1)
            loss1 = func.lovasz_softmax(predict1,geology)
            #loss2 = func.Dice_loss(geology,predict, beta=1, smooth = 1e-5)
            
            loss2 = func.xloss(predict,geology.squeeze().long())
            loss3 = 1 - ssim(pre, jiedian)
            L1_loss = nn.L1Loss()
            loss4  =  L1_loss(pre, jiedian)
            #loss2 = edge(out3, geo_edge) #+ edge(out2, geo_edge) + edge(out3, geo_edge)
            #loss3 = 1 - ssim(predict, geology)
            #loss3 = func._smooth_l1_loss(predict22,jiedian)
            #loss3 = func.huber(jiedian,predict22,1)
            loss = loss3+loss4*10+loss1+loss2
            #debug()
            # measure accuracy and record loss
            losses1.update(loss1.item(), observe.size(0)) 
            losses2.update(loss2.item(), observe.size(0)) 
            losses3.update(loss3.item(), observe.size(0))
            losses4.update(loss4.item(), observe.size(0))
            losses.update(loss.item(), observe.size(0))         
            predict3 = predict.data.max(1, keepdim=True)[1].byte()
            predict3 = predict3.to(torch.float32)
            correct += (predict3 == geology).sum().item()
            geology2 = geology.flatten(0)
            total1 += geology2.size(0)
            #predict4 = predict3.flatten(0)
            #c = (predict4 == geology2).squeeze()
            #for j in range(d):
                #geo = geology2[j]
                #class_correct[geo] += c[j].item()
                #class_total[geo] += 1
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % args_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, step, len(valid_loader), batch_time=batch_time, loss=losses))
                #print('Accuracy : %d %%' % (
    #100 * correct / total1))

            if step % 3 == 0:
                threadlist.append(threading.Thread(target=func.to_tensorboard, args=(lock, writer, torch.cat((jiedian, pre),0).cpu(), epoch*len(valid_loader)+step, 'img_valid', args_batch)))
                threadlist[-1].start()
        print('main thread finished, waiting for IO threads...')
        for thread in threadlist:
            thread.join()

        epoch_time = time.time() - epoch_end
        print('Valid time : {:.2f}min'.format(epoch_time/60))    
        writer.add_scalar('Loss1_Valid', losses1.avg, epoch)        
        writer.add_scalar('Loss2_Valid', losses2.avg, epoch)  
        writer.add_scalar('Loss3_Valid', losses3.avg, epoch)
        writer.add_scalar('Loss4_Valid', losses4.avg, epoch)
        writer.add_scalar('Loss_Valid', losses.avg, epoch)
           
    return losses.avg
    #for z in range(11):
        #print('Accuracy of %5s : %2d %%' % (
              #classes[z], 100 * class_correct[z] / class_total[z]))

min_loss = 10000
for epoch in range(args_epochs): 
    # train for one epoch
    train(train_loader, TomoNet, optimizer, scheduler, epoch, writer)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        current_loss = valid(valid_loader, TomoNet, epoch, writer)

    #if (epoch + 1) % (args_freq * 2) == 0:
    if min_loss>current_loss:
        min_loss = current_loss
        func.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': TomoNet.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, filename=save_path+'/minloss_checkpoint.pth.tar')

func.save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': TomoNet.state_dict(),
    'optimizer' : optimizer.state_dict(),
}, filename=save_path+'/lastcheckpoint.pth.tar')

