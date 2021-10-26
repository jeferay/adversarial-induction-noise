from logging import Logger
import time
from unicodedata import decomposition

from numpy.compat.py3k import asstr

import models
import torch
import torch.optim as optim
import util
import cw
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Evaluator():
    def __init__(self, data_loader, logger, config):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.acc_attack_meters = util.AverageMeter()
        self._accu_meters = util.AverageMeter()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        self.confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        
        
        return

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.acc_attack_meters = util.AverageMeter()
        self._accu_meters = util.AverageMeter()
        self.confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)#class-class矩阵，用于记录每个label所有sample被分类的结果的统计总和
        return

    def eval(self, epoch, model, target_labels = None,attack_type='pgd20'):#target_labels是一个映射，当其为none的时候，表示在原先label上的测试，否则为plus——one或者minus one
        model.eval()
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
            start = time.time()
            log_payload = self.eval_batch(images=images, labels=labels, model=model,target_labels = target_labels,attack_type=attack_type)
            end = time.time()
            time_used = end - start
        display = util.log_display(epoch=epoch,
                                   global_step=i,#已经遍历到了最后一个i
                                   time_elapse=time_used,
                                   **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        
        return

    def eval_with_save_res(self, model,res_save_type,attack_type='pgd20'):#此时要求dataloader已经不shuffle
        model.eval()
        res_targets=[]
        idx=0
        #确保没有shuffle
        for _, (images, labels) in enumerate(self.data_loader['train_dataset']):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch_size = images.shape[0]
            target_labels = torch.zeros_like(labels,dtype=torch.long,device=device)
            indices = self.eval_batch_with_save_res(images,labels,model,res_save_type,attack_type='pgd20')
            for i,(image,label) in enumerate(zip(images,labels)):
                for target in indices[i]:
                    if target!=label:
                        target_labels[i] = target
                        break
            assert((target_labels==labels).sum()==0)
            res_targets.append(target_labels)
        res_targets = torch.cat(res_targets,dim=0)

        assert(res_targets.shape[0]==50000)
        
        return res_targets


    

    def eval_batch_with_save_res(self,images,labels,model,res_save_type,attack_type='pgd20'):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            X_PGD = None
            if attack_type=='pgd20':
                X_PGD = self._pgd20_whitebox(model,images,labels,return_x_pgd=True)#得到x_pgd
            logits = model(X_PGD)
            if res_save_type=='MC':descending=True
            elif res_save_type=='LL':descending=False
            values,indices = torch.sort(logits,dim=1,descending=descending)
            return indices





    def eval_batch(self, images, labels, model,target_labels = None,attack_type='pgd20'):
        
        if target_labels!=None:labels = target_labels[labels]
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(images)
            
            loss = self.criterion(pred, labels)
            acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
            _, preds = torch.max(pred, 1)#返回index shape of batch
            """
            for t, p in zip(labels.view(-1), preds.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1
            """
            if attack_type=='pgd20':
                _acc,acc_attack = self._pgd20_whitebox(model,images,labels)#计算accpgd,单个batch,默认的是pdg20
            if attack_type=='cw':
                _acc,acc_attack = self._cw(model=model,x=images,y=labels,targeted=False)



        #累计计算结果
        self.loss_meters.update(loss.item(), n=images.size(0))#都是avg类
        self.acc_meters.update(acc.item(), n=images.size(0))
        self.acc5_meters.update(acc5.item(), n=images.size(0))
        self._accu_meters.update(_acc.item(),n=images.size(0))
        self.acc_attack_meters.update(acc_attack.item(),n=images.size(0))
        payload = {"acc": acc.item(),
                   "acc_avg": self.acc_meters.avg,
                   "acc5": acc5.item(),
                   "acc5_avg": self.acc5_meters.avg,
                   "loss": loss.item(),
                   "loss_avg": self.loss_meters.avg,
                   "_acc":_acc.item(),
                   "_acc_avg":self._accu_meters.avg,
                   attack_type:acc_attack.item(),
                   attack_type+"_avg":self.acc_attack_meters.avg
                   }
        return payload

    #如果return x_pgd，则返回的是x_pgd
    def _pgd20_whitebox(self, model, X, y, random_start=True, epsilon=0.031, num_steps=20, step_size=0.003,return_x_pgd=False):
        
        batch_size = y.size(0)
        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum(0)#求和表示相等的个数，但是此时还没有除以batchsize
        X_pgd = Variable(X.data, requires_grad=True)
        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)#得到了被攻击后的x
        
        if return_x_pgd:return X_pgd

        
        _,preds = torch.max(model(X_pgd),dim=1)#取argmax
        

        acc_pgd = (preds==y).float().sum(0)
        for t, p in zip(y.view(-1), preds.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1
        
        return acc.mul_(1/batch_size), acc_pgd.mul_(1/batch_size)#除以batch size

    def _cw(self,model,x,y,targeted=False):
        batch_size = y.size(0)
        model.eval()
        out = model(x)
        acc = (out.data.max(1)[1] == y.data).float().sum(0)#求和表示相等的个数，但是此时还没有除以batchsize
        
        adversary = cw.L2Adversary(targeted=False,
                           confidence=0.0,
                           search_steps=10,
                           #box=inputs_box,
                           optimizer_lr=5e-4)


        x_cw = adversary(model, x, y, to_numpy=False)
        assert isinstance(x_cw, torch.FloatTensor)
        assert x_cw.size() == x_cw.size()

        _,preds = torch.max(model(x_cw),dim=1)#取argmax
        
        acc_pgd = (preds==y).float().sum(0)
        
        return acc.mul_(1/batch_size), acc_pgd.mul_(1/batch_size)#除以batch size


    #def _pgd100_whitebox(self,model,X,y,random_start=True,epsilon=8/255, num_steps=100, step_size=)
