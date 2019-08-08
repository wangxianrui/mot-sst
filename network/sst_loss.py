# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import torch
import torch.nn.functional as F
from config import TrainConfig as Config


def print_grad(grad):
    print('hook:', grad)


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
        # test
        # predict.register_hook(print_grad)
        #
        predict_pre = torch.nn.Softmax(dim=3)(predict[:, :, :-1, :])
        predict_next = torch.nn.Softmax(dim=2)(predict[:, :, :, :-1])
        predict_union = (predict_pre[:, :, :, :-1] + predict_next[:, :, :-1, :]) / 2

        target = mask * target
        target_pre = target[:, :, :-1, :]
        target_next = target[:, :, :, :-1]
        target_union = target[:, :, :-1, :-1]

        target_pre_num = target_pre.sum()
        target_next_num = target_next.sum()
        target_union_num = target_union.sum()

        # loss 4 part
        loss_pre = FocalLoss()(predict_pre, target_pre, target_pre_num)
        loss_next = FocalLoss()(predict_next, target_next, target_next_num)
        loss_union = FocalLoss()(predict_union, target_union, target_union_num)
        loss_unmatched = FocalLoss()(predict_pre[:, :, :, -1], target_pre[:, :, :, -1], target_pre_num - target_union_num) \
                         + FocalLoss()(predict_next[:, :, -1, :], target_next[:, :, -1, :], target_next_num - target_union_num)
        loss_sim = 2 * (target_union * torch.abs(predict_pre[:, :, :, :-1] - predict_next[:, :, :-1, :])).sum() / target_union_num
        loss = (loss_pre + loss_next + loss_union + loss_sim) / 4.0
        return loss_pre, loss_next, loss_union, loss_sim, loss, target_pre_num, target_next_num, target_union_num


class FocalLoss(object):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.alpha = Config.focal_alpha
        self.gamma = Config.focal_gamma

    def __call__(self, predict, target, target_num):
        if target_num == 0:
            return torch.tensor(0, device=predict.device)
        return -self.alpha * (target * torch.pow(1 - predict, self.gamma) * torch.log(predict)).sum() / target_num
