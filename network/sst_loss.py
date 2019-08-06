# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import torch
import torch.nn.functional as F
from config import Config


class SSTLoss(object):
    def __init__(self):
        super(SSTLoss, self).__init__()
        self.max_object = Config.max_object

    def __call__(self, predict, target, mask0, mask1):
        """
        :param predict: b, 1, N+1, N+1
        :param target: b, 1, N+1, N+1
        :param mask0: pre_mask b, 1, N+1
        :param mask1: next_mask b, 1, N+1
        :return:
        mask, predict, target:  N+1, N+1
        *_pre:  N, N+1
        *_next: N+1, N
        *_union: N, N
        """
        mask = mask0.unsqueeze(3) * mask1.unsqueeze(2)

        predict = mask * predict
        predict_pre = torch.nn.Softmax(dim=3)(predict[:, :, :-1, :])
        predict_next = torch.nn.Softmax(dim=2)(predict[:, :, :, :-1])
        predict_union = (predict_pre[:, :, :, :-1] + predict_next[:, :, :-1, :]) / 2

        target = mask * target
        target_pre = target[:, :, :-1, :]
        target_next = target[:, :, :, :-1]
        target_union = target[:, :, :-1, :-1]

        # mask = (mask0.unsqueeze(3) * mask1.unsqueeze(2))
        # mask_pre = mask[:, :, :-1, :].clone()
        # mask_next = mask[:, :, :, :-1].clone()
        # mask_union = mask[:, :, :-1, :-1].clone()
        #
        # predict_pre = predict[:, :, :-1, :].clone()
        # predict_pre = torch.nn.Softmax(dim=3)(mask_pre * predict_pre)
        # predict_next = predict[:, :, :, :-1].clone()
        # predict_next = torch.nn.Softmax(dim=2)(mask_next * predict_next)
        # # predict_union = torch.max(predict_pre[:, :, :, :-1], predict_next[:, :, :-1, :])
        # predict_union = (predict_pre[:, :, :, :-1] + predict_next[:, :, :-1, :]) / 2
        #
        # target_pre = mask_pre * target[:, :, :-1, :]
        # target_next = mask_next * target[:, :, :, :-1]
        # target_union = mask_union * target[:, :, :-1, :-1]

        target_pre_num = target_pre.sum()
        target_next_num = target_next.sum()
        target_union_num = target_union.sum()

        # loss 4 part
        assert target_pre_num > 0, 'target_pre_num should > 0'
        # loss_pre = -(target_pre * torch.log(predict_pre)).sum() / target_pre_num
        loss_pre = FocalLoss()(predict_pre, target_pre, target_pre_num)

        assert target_next_num > 0, 'traget_next_num should > 0'
        # loss_next = -(target_next * torch.log(predict_next)).sum() / target_next_num
        loss_next = FocalLoss()(predict_next, target_next, target_next_num)

        assert target_union_num > 0, 'target_union_num should > 0'
        # loss_union = -(target_union * torch.log(predict_union)).sum() / target_union_num
        loss_union = FocalLoss()(predict_union, target_union, target_union_num)
        loss_sim = 2 * (target_union * torch.abs(predict_pre[:, :, :, :-1] - predict_next[:, :, :-1, :])).sum() / target_union_num

        loss = (loss_pre + loss_next + loss_union + loss_sim) / 4.0
        return loss_pre, loss_next, loss_union, loss_sim, loss


class FocalLoss(object):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.alpha = 1
        self.gamma = 2

    def __call__(self, predict, target, target_num):
        return -self.alpha * (target * torch.pow(1 - predict, self.gamma) * torch.log(predict)).sum() / target_num
