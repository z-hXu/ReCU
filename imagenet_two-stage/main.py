import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models_imagenet
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from utils import *
from modules import *
from datetime import datetime 
import dataset
import torchvision.models as models


def main():
    global args, best_prec1, best_prec5, conv_modules
    best_prec1 = 0
    best_prec5 = 0

    args.print_freq=int(256/args.batch_size*500)

    if args.evaluate:
        args.results_dir = '/tmp'
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not args.resume and not args.evaluate:
        with open(os.path.join(save_path,'config.txt'), 'w') as args_file:
            args_file.write(str(datetime.now())+'\n\n')
            for args_n,args_v in args.__dict__.items():
                args_v = '' if not args_v and not isinstance(args_v,int) else args_v
                args_file.write(str(args_n)+':  '+str(args_v)+'\n')

        setup_logging(os.path.join(save_path, 'logger.log'))
        logging.info("saving to %s", save_path)
        logging.info("run arguments: %s", args)
    else: 
        setup_logging(os.path.join(save_path, 'logger.log'), filemode='a')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        if args.seed > 0:
            set_seed(args.seed) 
        else: 
            cudnn.benchmark = True
    else:
        args.gpus = None

    if args.dataset=='tinyimagenet':
        num_classes=200
        model_zoo = 'models_imagenet.'
    elif args.dataset=='imagenet':
        num_classes=1000
        model_zoo = 'models_imagenet.'
    elif args.dataset=='cifar10': 
        num_classes=10
        model_zoo = 'models_cifar.'
    elif args.dataset=='cifar100': 
        num_classes=100
        model_zoo = 'models_cifar.'

    #* create model
    if len(args.gpus)==1:
        model = eval(model_zoo+args.model)(num_classes=num_classes).cuda()
        if args.teacher:
            teacher = eval('models.'+args.teacher)(pretrained=True).cuda()
    else: 
        model = nn.DataParallel(eval(model_zoo+args.model)(num_classes=num_classes))
        if args.teacher:
            teacher = nn.DataParallel(eval('models.'+args.teacher)(pretrained=True))
    if args.teacher:
        teacher = teacher.type(args.type)
        teacher.eval()
    else:
        teacher = None
    if not args.resume:
        logging.info("creating model %s", args.model)
        logging.info("model structure: ")
        for name,module in model._modules.items():
            logging.info('\t'+str(name)+': '+str(module))
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("number of parameters: %d", num_parameters)
        logging.info("load teacher model: %s", args.teacher)
        if args.stage1:
            logging.info("load stage1 model: %s", args.stage1)
            model.load_state_dict(dataset.add_module_fromdict(torch.load(args.stage1, map_location=torch.device('cpu'))['state_dict']), strict=False)

    #* evaluate
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            logging.error('invalid checkpoint: {}'.format(args.evaluate))
            return 
        else: 
            checkpoint = torch.load(args.evaluate, map_location=torch.device('cpu'))
            if len(args.gpus)>1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                        args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = os.path.join(save_path,'checkpoint.pth.tar')
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            if len(args.gpus)>1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            best_prec5 = checkpoint['best_prec5']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion = criterion.type(args.type)
    if args.teacher:
        criterionKD = DistributionLoss().type(args.type)
    else:
        criterionKD = None
    model = model.type(args.type)

    if args.evaluate:
        if args.use_dali:
            val_loader = dataset.get_imagenet(
                        type='val',
                        image_dir=args.data_path,
                        batch_size=args.batch_size_test,
                        num_threads=args.workers,
                        crop=224,
                        device_id='cuda:0',
                        num_gpus=1)
        else:
            val_loader = dataset.get_imagenet_torch(
                        type='val',
                        image_dir=args.data_path,
                        batch_size=args.batch_size_test,
                        num_threads=args.workers,
                        device_id='cuda:0'
                        )
        with torch.no_grad():
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0)
        logging.info('\n Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
        return

    #* load dataset
    if args.dataset=='imagenet':
        if args.use_dali:
            train_loader = dataset.get_imagenet(
                            type='train',
                            image_dir=args.data_path,
                            batch_size=args.batch_size,
                            num_threads=args.workers,
                            crop=224,
                            device_id='cuda:0',
                            num_gpus=1)
            val_loader = dataset.get_imagenet(
                            type='val',
                            image_dir=args.data_path,
                            batch_size=args.batch_size_test,
                            num_threads=args.workers,
                            crop=224,
                            device_id='cuda:0',
                            num_gpus=1)
        else:
            train_loader = dataset.get_imagenet_torch(
                            type='train',
                            image_dir=args.data_path,
                            batch_size=args.batch_size,
                            num_threads=args.workers,
                            device_id='cuda:0',
                            )
            val_loader = dataset.get_imagenet_torch(
                            type='val',
                            image_dir=args.data_path,
                            batch_size=args.batch_size_test,
                            num_threads=args.workers,
                            device_id='cuda:0'
                            )             
    else: 
        train_loader, val_loader = dataset.load_data(
                                    dataset=args.dataset,
                                    data_path=args.data_path,
                                    batch_size=args.batch_size,
                                    batch_size_test=args.batch_size_test,
                                    num_workers=args.workers)

    conv_param = (param for name, param in model.named_parameters() if ('fc' in name or 'conv' in name))
    param = (param for name, param in model.named_parameters() if not ('fc' in name or 'conv' in name))
    #* optimizer settings
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params':conv_param,'initial_lr':args.lr},
                                     {'params':param,'initial_lr':args.lr,'weight_decay':0.}],
                                    lr=args.lr, 
                                    momentum=args.momentum, 
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params':conv_param,'initial_lr':args.lr},
                                      {'params':param,'initial_lr':args.lr,'weight_decay':0.}],
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay) 
    else: 
        logging.error("Optimizer '%s' not defined.", args.optimizer)

    if args.lr_type == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warm_up*4, eta_min = 0, last_epoch=args.start_epoch)
    elif args.lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1, last_epoch=-1)
    elif args.lr_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (1.0-(epoch-args.warm_up*4)/(args.epochs-args.warm_up*4)), last_epoch=-1)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        logging.info("criterion: %s", criterion)
        logging.info('scheduler: %s', lr_scheduler)

    def cpt_tau(epoch):
        "compute tau"
        a = torch.tensor(np.e)
        T_min, T_max = torch.tensor(args.tau_min).float(), torch.tensor(args.tau_max).float()
        A = (T_max - T_min) / (a - 1)
        B = T_min - A
        tau = A * torch.tensor([torch.pow(a, epoch/args.epochs)]).float() + B
        return tau 

    #* record names of conv_modules
    conv_modules=[]
    for name, module in model.named_modules():
        if isinstance(module,BinarizeConv2d):
            conv_modules.append(module)


    for epoch in range(args.start_epoch+1, args.epochs):
        time_start = datetime.now()
        #* warm up
        if args.warm_up and epoch < 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch+1) / 5
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])
            break

        #* compute threshold tau
        tau = cpt_tau(epoch)
        for module in conv_modules:
            module.tau = tau.cuda()

        #* training
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer, teacher, criterionKD)

        #* adjust Lr
        if epoch >= 4 * args.warm_up:
            lr_scheduler.step()

        #* evaluating
        with torch.no_grad():
            val_loss, val_prec1, val_prec5 = validate(
                val_loader, model, criterion, epoch)

        #* remember best prec
        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = max(val_prec1, best_prec1)
            best_prec5 = max(val_prec5, best_prec5)
            best_epoch = epoch
            best_loss = val_loss

        #* save model
        if epoch % 1 == 0:
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            model_optimizer = optimizer.state_dict()
            model_scheduler = lr_scheduler.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model_state_dict,
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                'optimizer': model_optimizer,
                'lr_scheduler': model_scheduler,
            }, is_best, path=save_path)

        if args.time_estimate > 0 and epoch % args.time_estimate==0:
            time_end = datetime.now()
            cost_time,finish_time = get_time(time_end-time_start,epoch,args.epochs)
            logging.info('Time cost: '+cost_time+'\t'
                        'Time of Finish: '+finish_time)

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))


    logging.info('*'*50+'DONE'+'*'*50)
    logging.info('\n Best_Epoch: {0}\t'
                     'Best_Prec1 {prec1:.4f} \t'
                     'Best_Prec5 {prec5:.4f} \t'
                     'Best_Loss {loss:.3f} \t'
                     .format(best_epoch+1, prec1=best_prec1, prec5=best_prec5, loss=best_loss))


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, teacher=None, criterionKD=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    if args.use_dali:
        for i, batch_data in enumerate(data_loader):
            #* measure data loading time
            data_time.update(time.time() - end)
            inputs = batch_data[0]['data']
            target = batch_data[0]['label'].squeeze().long()
            batchsize = args.batch_size if training else args.batch_size_test
            len_dataloader = int(np.ceil(data_loader._size/batchsize))
            if args.gpus is not None:
                inputs = inputs.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            input_var = Variable(inputs.type(args.type))
            target_var = Variable(target)

            #* compute output
            output = model(input_var)
            if not training or not args.teacher:
                loss = criterion(output, target_var)

            if type(output) is list:
                output = output[0]

            if training:
                if args.teacher:
                    with torch.no_grad():
                        output_t = teacher(input_var)
                    loss = criterionKD(output, output_t)
                #* back-propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #* measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            #* measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, i * batchsize, data_loader._size,
                                phase='TRAINING' if training else 'EVALUATING',
                                batch_time=batch_time,
                                data_time=data_time, loss=losses, 
                                top1=top1, top5=top5))
    else:
        for i, (inputs, target) in enumerate(data_loader):
            #* measure data loading time
            data_time.update(time.time() - end)
            if args.gpus is not None:
                inputs = inputs.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            input_var = Variable(inputs.type(args.type))
            target_var = Variable(target)
            batchsize = args.batch_size if training else args.batch_size_test

            #* compute output
            output = model(input_var)
            if not training or not teacher:
                loss = criterion(output, target_var)

            if type(output) is list:
                output = output[0]

            if training:
                if args.teacher:
                    with torch.no_grad():
                        output_t = teacher(input_var)
                    loss = criterionKD(output, output_t)
                #* back-propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #* measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            #* measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, i * batchsize, len(data_loader) * batchsize,
                                phase='TRAINING' if training else 'EVALUATING',
                                batch_time=batch_time,
                                data_time=data_time, loss=losses,
                                top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, teacher, criterionKD):
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer, teacher=teacher, criterionKD=criterionKD)


def validate(data_loader, model, criterion, epoch):
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
