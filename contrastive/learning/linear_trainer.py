from __future__ import print_function

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict

from .util import AverageMeter, accuracy, xray_accuracy
from .base_trainer import BaseTrainer


class LinearTrainer(BaseTrainer):
    """trainer for Linear evaluation"""
    def __init__(self, args):
        super(LinearTrainer, self).__init__(args)

    def logging(self, epoch, logs, lr=None, train=True):
        """ logging to tensorboard

        Args:
          epoch: training epoch
          logs: loss and accuracy
          lr: learning rate
          train: True of False
        """
        args = self.args
        if args.rank == 0:
            pre = 'train_' if train else 'test_'
            self.logger.log_value(pre+'acc', logs[0], epoch)
            self.logger.log_value(pre+'acc5', logs[1], epoch)
            self.logger.log_value(pre+'loss', logs[2], epoch)
            if train and (lr is not None):
                self.logger.log_value('learning_rate', lr, epoch)

    def wrap_up(self, model, classifier):
        """Wrap up models with DDP

        Args:
          model: pretrained encoder, should be frozen
          classifier: linear classifier
        """
        args = self.args
        model = model.cuda()
        classifier = classifier.cuda()
        model.eval()
        model = DDP(model, device_ids=[args.gpu])
        classifier = DDP(classifier, device_ids=[args.gpu])

        return model, classifier

    def load_encoder_weights(self, model):
        """load pre-trained weights for encoder

        Args:
          model: pretrained encoder, should be frozen
        """
        args = self.args
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location='cpu')
            state_dict = ckpt['model']
            if args.modal == 'RGB':
                # Unimodal (RGB) case
                encoder_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    if 'encoder' in k:
                        k = k.replace('encoder.', '')
                        encoder_state_dict[k] = v
                model.encoder.load_state_dict(encoder_state_dict)
            else:
                # Multimodal (CMC) case
                encoder1_state_dict = OrderedDict()
                encoder2_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    if 'encoder1' in k:
                        k = k.replace('encoder1.', '')
                        encoder1_state_dict[k] = v
                    if 'encoder2' in k:
                        k = k.replace('encoder2.', '')
                        encoder2_state_dict[k] = v
                model.encoder1.load_state_dict(encoder1_state_dict)
                model.encoder2.load_state_dict(encoder2_state_dict)
            print('Pre-trained weights loaded!')
        else:
            print('==============================')
            print('warning: no pre-trained model!')
            print('==============================')

        return model

    def resume_model(self, classifier, optimizer):
        """load classifier checkpoint"""
        args = self.args
        start_epoch = 1
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint['epoch'] + 1
                classifier.load_state_dict(checkpoint['classifier'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        return start_epoch

    def save(self, classifier, optimizer, epoch):
        """save classifier to checkpoint"""
        args = self.args
        if args.local_rank == 0:
            # saving the classifier to each instance
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(args.model_folder, 'current.pth')
            torch.save(state, save_file)
            if epoch % args.save_freq == 0:
                save_file = os.path.join(
                    args.model_folder, 'ckpt_epoch_{}.pth'.format(epoch))
                torch.save(state, save_file)
                # help release GPU memory
            del state

    def train(self, epoch, train_loader, model, classifier,
              criterion, optimizer):
        time1 = time.time()
        args = self.args

        activation = nn.Sigmoid() if args.loss == 'bce' else None

        if not args.finetune:
            model.eval()
        else:
            model.train()
        classifier.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        metric1 = AverageMeter()
        metric2 = AverageMeter()
        metric3 = AverageMeter()

        end = time.time()
        for idx, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            input = input.float()
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # forward
            if not args.finetune:
                with torch.no_grad():
                    feat = model(x=input, mode=2)
                    feat = feat.detach()
            else:
                feat = model(x=input, mode=2)

            output = classifier(feat)
            if activation is not None:
                output = activation(output)
            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))
            if activation is not None:
                aucs, f1s, accs = xray_accuracy(output, target)
                metric1.update(np.mean(aucs), input.size(0))
                metric2.update(np.mean(f1s), input.size(0))
                metric3.update(np.mean(accs), input.size(0))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                metric1.update(acc1[0], input.size(0))
                metric2.update(acc5[0], input.size(0))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0 and idx % args.print_freq == 0:
                if activation is not None:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'AUC {metric1.val:.3f} ({metric1.avg:.3f})\t'
                          'F1 {metric2.val:.3f} ({metric2.avg:.3f})\t'
                          'Acc {metric3.val:.3f} ({metric3.avg:.3f})'.format(
                           epoch, idx, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses, metric1=metric1, metric2=metric2, metric3=metric3))
                    sys.stdout.flush()
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {metric1.val:.3f} ({metric1.avg:.3f})\t'
                          'Acc@5 {metric2.val:.3f} ({metric2.avg:.3f})'.format(
                           epoch, idx, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses, metric1=metric1, metric2=metric2))
                    sys.stdout.flush()

        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        return metric1.avg, metric2.avg, losses.avg

    def validate(self, epoch, val_loader, model, classifier, criterion):
        time1 = time.time()
        args = self.args

        model.eval()
        classifier.eval()

        activation = nn.Sigmoid() if args.loss == 'bce' else None

        batch_time = AverageMeter()
        losses = AverageMeter()
        metric1 = AverageMeter()
        metric2 = AverageMeter()
        metric3 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for idx, (input, target) in enumerate(val_loader):
                input = input.float()
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                feat = model(x=input, mode=2)
                output = classifier(feat)
                if activation is not None:
                    output = activation(output)
                loss = criterion(output, target)

                losses.update(loss.item(), input.size(0))
                if activation is not None:
                    aucs, f1s, accs = xray_accuracy(output, target)
                    metric1.update(np.mean(aucs), input.size(0))
                    metric2.update(np.mean(f1s), input.size(0))
                    metric3.update(np.mean(accs), input.size(0))
                else:
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    metric1.update(acc1[0], input.size(0))
                    metric2.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if args.local_rank == 0 and idx % args.print_freq == 0:
                    if activation is not None:
                        print('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'AUC {metric1.val:.3f} ({metric1.avg:.3f})\t'
                              'F1 {metric2.val:.3f} ({metric2.avg:.3f})\t'
                              'Acc {metric3.val:.3f} ({metric3.avg:.3f})'.format(
                               epoch, idx, len(val_loader), batch_time=batch_time,
                               loss=losses, metric1=metric1, metric2=metric2, metric3=metric3))
                        sys.stdout.flush()
                    else:
                        print('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Acc@1 {metric1.val:.3f} ({metric1.avg:.3f})\t'
                              'Acc@5 {metric2.val:.3f} ({metric2.avg:.3f})'.format(
                               epoch, idx, len(val_loader), batch_time=batch_time,
                               data_time=data_time, loss=losses, metric1=metric1, metric2=metric2))
                        sys.stdout.flush()

            if activation is not None:
                print(' * AUC {metric1.avg:.3f} F1 {metric2.avg:.3f} Acc {metric2.avg:.3f}'
                      .format(metric1=metric1, metric2=metric2))
            else:
                print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=metric1, top5=metric2))

        time2 = time.time()
        print('eval epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        return metric1.avg, metric2.avg, losses.avg
