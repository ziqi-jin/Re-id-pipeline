import time
import logging
import copy
import torch
import torch.nn as nn
import numpy as np
from models.models import PCB_test
from torch.autograd import Variable
import os
logger = logging.getLogger(__name__)
from tqdm import tqdm

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet
        # compute output
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        if not config.MODEL.PCB:
            _,preds = torch.max(output.data,1)
            loss = criterion(output, target)
        else:
            part = {}
            sm = nn.Softmax(dim=1)
            num_part = 6
            for i in range(num_part):
                part[i] = output[i]
            score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
            _,preds = torch.max(score.data,1)
            
            loss = criterion(part[0], target)
            for i in range(num_part-1):
                loss += criterion(part[i+1], target)


        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if not config.MODEL.PCB:
            prec1, prec5 = accuracy(output, target, (1, 5))
        else:
            prec1, prec5 = accuracy(score, target, (1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    # model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # compute output
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            if not config.MODEL.PCB:
                loss = criterion(output, target)
            else:
                part = {}
                sm = nn.Softmax(dim=1)
                num_part = 6
                for i in range(num_part):
                    part[i] = output[i]     
                score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                loss = criterion(part[0], target)
                for i in range(num_part-1):
                    loss += criterion(part[i+1], target)           
            # target = target.cuda(non_blocking=True)


            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            if not config.MODEL.PCB:
                prec1, prec5 = accuracy(output, target, (1, 5))
            else:
                prec1, prec5 = accuracy(score, target, (1, 5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Error@5 {error5:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t' \
              'Accuracy@5 {top5.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, top1=top1, top5=top5,
                  error1=100-top1.avg, error5=100-top5.avg)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return top1.avg

def evaluate(cfg,model,query_loader,gallery_loader,query_cam,query_label,gallery_cam,gallery_label):
    """
    this function we get a model,and we use the query prope to query a list form the gallery.
    we will use mAP, and cmc score .
    :input:
        model:the model need delete the final linear layer
        result:a dict include the featrue ,camera, information and label from query and gallery
    :return: mAP and cmc score
    """
    #first we need to remove the last layer of model
    # print(model)
    global config
    config = cfg

    model_modify = copy.deepcopy(model)
    if config.MODEL.PCB:
        print(model_modify)
        print('##############################################')
        model_modify = PCB_test(model_modify)
    else:
        model_modify.module.classifier.classifier = torch.nn.Sequential()
    model_modify.eval()
    model_modify = model_modify.cuda()

    with torch.no_grad():
        gallery_feature = extract_feature(model_modify, gallery_loader)
        query_feature = extract_feature(model_modify, query_loader)

    # print(query_feature.shape)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    flag=0
    for i in tqdm(range(len(query_label))):
        ap_tmp, CMC_tmp = cal_ap_cmc(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        #计数
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
    return CMC[0]/len(query_label)

def cal_ap_cmc(query_featrue, query_label, query_cam, gallery_feature, gallery_label, gallery_cam):
    query = query_featrue.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gallery_feature, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    #gallerlabel and query label 应该是等长的 ，tensor 和ndarray 都可以使用np.argwhere,因为label是个list
    query_index = np.argwhere(np.array(gallery_label) == query_label)
    camera_index = np.argwhere(np.array(gallery_cam) == query_cam)

    # setdiff1d即找出不同camera 中的正确的index，相同的不算good index，为什么呢
    # print(query_index,camera_index,'qeruy index and carmera')
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(np.array(gallery_label) == -1)
    # intersect1d 返回相同元素
    junk_index2 = np.intersect1d(query_index, camera_index)

    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP_cmc(index, good_index, junk_index)
    return CMC_tmp

def compute_mAP_cmc(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    #只要包含就算，因此用 rowsgood的第一个也就是最前面的对的那一个的index
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        outputs = torch.FloatTensor(n,512).zero_().cuda()
        if config.MODEL.PCB:
            outputs = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts
        
        #this 512 should be modified as a config parameter
        # this part include flip img and
        # ff = torch.FloatTensor(n,512).zero_().cuda()
        # for i in range(2):
        #     if(i==1):
        #         img = fliplr(img)
        #     input_img = Variable(img.cuda())
        #     for scale in ms:
        #         if scale != 1:
        #             # bicubic is only  available in pytorch>= 1.1
        #             input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
        #         outputs = model(input_img)
        #         ff += outputs
        input_img = Variable(img.cuda())
        outputs = model(input_img)

        # norm feature

        #归一化  a.b = |a|*|b|cos(theta) if norm(a) ,|a|=1,so a.b=cos(theta)
        if config.MODEL.PCB:
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)*np.sqrt(6)
            outputs = outputs.div(fnorm.expand_as(outputs))
            outputs = outputs.view(outputs.size(0),-1)
        else:
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            outputs = outputs.div(fnorm.expand_as(outputs))

        features = torch.cat((features,outputs.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        #val 是平均的
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
