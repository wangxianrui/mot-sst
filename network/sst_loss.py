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
        mask = (mask0.unsqueeze(3) * mask1.unsqueeze(2))
        mask_pre = mask[:, :, :-1, :].clone()
        mask_next = mask[:, :, :, :-1].clone()
        mask_union = mask[:, :, :-1, :-1].clone()

        predict_pre = predict[:, :, :-1, :].clone()
        predict_pre = torch.nn.Softmax(dim=3)(mask_pre * predict_pre)
        predict_next = predict[:, :, :, :-1].clone()
        predict_next = torch.nn.Softmax(dim=2)(mask_next * predict_next)
        predict_union = torch.max(predict_pre[:, :, :, :-1], predict_next[:, :, :-1, :])

        target_pre = mask_pre * target[:, :, :-1, :]
        target_next = mask_next * target[:, :, :, :-1]
        target_union = mask_union * target[:, :, :-1, :-1]

        target_pre_num = target_pre.sum()
        target_next_num = target_next.sum()
        target_union_num = target_union.sum()

        # TODO ensure num > 0
        # loss 4 part
        assert target_pre_num > 0, 'target_pre_num should > 0'
        loss_pre = -(target_pre * torch.log(predict_pre)).sum() / target_pre_num

        assert target_next_num > 0, 'traget_next_num should > 0'
        loss_next = -(target_next * torch.log(predict_next)).sum() / target_next_num

        assert target_union_num > 0, 'target_union_num should > 0'
        loss_union = -(target_union * torch.log(predict_union)).sum() / target_union_num
        loss_sim = (target_union * torch.abs(predict_pre[:, :, :, :-1] - predict_next[:, :, :-1, :])).sum() / target_union_num

        loss = (loss_pre + loss_next + loss_union + loss_sim) / 4.0
        return loss_pre, loss_next, loss_union, loss_sim, loss
