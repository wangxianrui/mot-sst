import torch
import torch.nn.functional as F
from config import Config


class SSTLoss(torch.nn.Module):
    def __init__(self, use_gpu=Config.use_cuda):
        super(SSTLoss, self).__init__()
        self.use_gpu = use_gpu
        self.max_object = Config.max_object

    def forward(self, predict, target, mask0, mask1):
        mask_pre = mask0[:, :, :]
        mask_next = mask1[:, :, :]
        mask0 = mask0.unsqueeze(3)
        mask1 = mask1.unsqueeze(2)
        if self.use_gpu:
            mask0 = mask0.cuda()
            mask1 = mask1.cuda()

        mask_region = (mask0 * mask1).float()
        mask_region_pre = mask_region.clone()  # note: should use clone (fix this bug)
        mask_region_pre[:, :, -1, :] = 0
        mask_region_next = mask_region.clone()  # note: should use clone (fix this bug)
        mask_region_next[:, :, :, -1] = 0
        mask_region_union = mask_region_pre * mask_region_next

        predict_pre = torch.nn.Softmax(dim=3)(mask_region_pre * predict)
        predict_next = torch.nn.Softmax(dim=2)(mask_region_next * predict)
        predict_all = predict_pre.clone()
        predict_all[:, :, :self.max_object, :self.max_object] = \
            torch.max(predict_pre, predict_next)[:, :, :self.max_object, :self.max_object]

        target = target.float()
        target_pre = mask_region_pre * target
        target_next = mask_region_next * target
        target_union = mask_region_union * target
        target_num = target.sum()
        target_num_pre = target_pre.sum()
        target_num_next = target_next.sum()
        target_num_union = target_union.sum()

        # todo: remove the last row negative effect
        if int(target_num_pre.item()):
            loss_pre = - (target_pre * torch.log(predict_pre)).sum() / target_num_pre
        else:
            loss_pre = - (target_pre * torch.log(predict_pre)).sum()
        if int(target_num_next.item()):
            loss_next = - (target_next * torch.log(predict_next)).sum() / target_num_next
        else:
            loss_next = - (target_next * torch.log(predict_next)).sum()
        if int(target_num_pre.item()) and int(target_num_next.item()):
            loss = -(target_pre * torch.log(predict_all)).sum() / target_num_pre
        else:
            loss = -(target_pre * torch.log(predict_all)).sum()
        if int(target_num_union.item()):
            loss_similarity = (target_union * (torch.abs((1 - predict_pre) - (1 - predict_next)))).sum() / target_num
        else:
            loss_similarity = (target_union * (torch.abs((1 - predict_pre) - (1 - predict_next)))).sum()

        _, indexes_ = target_pre.max(dim=3)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_pre = predict_all.max(dim=3)
        indexes_pre = indexes_pre[:, :, :-1]
        mask_pre_num = mask_pre[:, :, :-1].sum().item()
        if mask_pre_num:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[
                mask_pre[:, :, :-1]]).float().sum() / mask_pre_num
        else:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]).float().sum() + 1

        _, indexes_ = target_next.max(2)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_next = predict_next.max(2)
        indexes_next = indexes_next[:, :, :-1]
        mask_next_num = mask_next[:, :, :-1].sum().item()
        if mask_next_num:
            accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[
                mask_next[:, :, :-1]]).float().sum() / mask_next_num
        else:
            accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() + 1

        return loss_pre, loss_next, loss_similarity, \
               (loss_pre + loss_next + loss + loss_similarity) / 4.0, accuracy_pre, accuracy_next, (
                       accuracy_pre + accuracy_next) / 2.0, indexes_pre
