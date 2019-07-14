import torch
import torch.nn.functional as F
from config import Config


class SSTLoss(torch.nn.Module):
    def __init__(self):
        super(SSTLoss, self).__init__()
        self.max_object = Config.max_object

    def forward(self, predict, target, mask0, mask1):
        '''
        @description: 
        @param {
            param predict: b, 1, N+1, N+1
            param target: b, 1, N+1, N+1
            param mask0: pre mask b, 1, N+1
            param mask1: next mask b, 1, N+1
        } 
        @return: 
         mask, predict, target:  N+1, N+1
        *_pre:  N, N+1
        *_next: N+1, N
        *_union: N, N
        '''
        mask = (mask0.unsqueeze(3) * mask1.unsqueeze(2))
        mask_pre = mask[:, :, :-1, :].clone()
        mask_next = mask[:, :, :, :-1].clone()
        mask_union = mask[:, :, :-1, :-1].clone()
        mask_pre_num = mask_pre.sum()
        mask_next_num = mask_next.sum()
        mask_union_num = mask_union.sum()

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

        # loss 4 part
        if target_pre_num > 0:
            loss_pre = -(target_pre * torch.log(predict_pre)).sum() / target_pre_num
        else:
            loss_pre = 0
        if target_next_num > 0:
            loss_next = -(target_next * torch.log(predict_next)).sum() / target_next_num
        else:
            loss_next = 0
        if target_union_num > 0:
            loss_union = -(target_union * torch.log(predict_union)).sum() / target_union_num
            loss_sim = (target_union * torch.abs(predict_pre[:, :, :, :-1] - predict_next[:, :, :-1, :])).sum() / target_union_num
        else:
            loss_union = 0
            loss_sim = 0
        loss = (loss_pre + loss_next + loss_union + loss_sim) / 4.0

        # accuracy
        _, target_pre_index = target_pre.max(dim=2)
        _, predict_pre_index = predict_pre.max(dim=2)
        if mask_pre_num > 0:
            accuracy_pre = (target_pre_index == predict_pre_index).sum() / mask_pre_num
        else:
            accuracy_pre = 0

        _, target_next_index = target_next.max(dim=3)
        _, predict_next_index = predict_next.max(dim=3)
        if mask_next_num > 0:
            accuracy_next = (target_next_index == predict_next_index).sum() / mask_next_num
        else:
            accuracy_next = 0
        accuracy = (accuracy_pre + accuracy_next) / 2

        return loss_pre, loss_next, loss_union, loss_sim, loss, accuracy_pre, accuracy_next, accuracy
