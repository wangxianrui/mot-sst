import torch
import torch.nn.functional as F
from config import Config


class SSTLoss(torch.nn.Module):
    def __init__(self, use_gpu=Config.use_cuda):
        super(SSTLoss, self).__init__()
        self.use_gpu = use_gpu
        self.max_object = Config.max_object

    def forward(self, input, target, mask0, mask1):
        mask_pre = mask0[:, :, :]
        mask_next = mask1[:, :, :]
        mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, self.max_object + 1)
        mask1 = mask1.unsqueeze(2).repeat(1, 1, self.max_object + 1, 1)
        if self.use_gpu:
            mask0 = mask0.cuda()
            mask1 = mask1.cuda()

        mask_region = (mask0 * mask1).float()
        mask_region_pre = mask_region.clone()  # note: should use clone (fix this bug)
        mask_region_pre[:, :, self.max_object, :] = 0
        mask_region_next = mask_region.clone()  # note: should use clone (fix this bug)
        mask_region_next[:, :, :, self.max_object] = 0
        mask_region_union = mask_region_pre * mask_region_next

        input_pre = torch.nn.Softmax(dim=3)(mask_region_pre * input)
        input_next = torch.nn.Softmax(dim=2)(mask_region_next * input)
        input_all = input_pre.clone()
        input_all[:, :, :self.max_object, :self.max_object] = \
            torch.max(input_pre, input_next)[:, :, :self.max_object, :self.max_object]

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
            loss_pre = - (target_pre * torch.log(input_pre)).sum() / target_num_pre
        else:
            loss_pre = - (target_pre * torch.log(input_pre)).sum()
        if int(target_num_next.item()):
            loss_next = - (target_next * torch.log(input_next)).sum() / target_num_next
        else:
            loss_next = - (target_next * torch.log(input_next)).sum()
        if int(target_num_pre.item()) and int(target_num_next.item()):
            loss = -(target_pre * torch.log(input_all)).sum() / target_num_pre
        else:
            loss = -(target_pre * torch.log(input_all)).sum()
        if int(target_num_union.item()):
            loss_similarity = (target_union * (torch.abs((1 - input_pre) - (1 - input_next)))).sum() / target_num
        else:
            loss_similarity = (target_union * (torch.abs((1 - input_pre) - (1 - input_next)))).sum()

        _, indexes_ = target_pre.max(dim=3)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_pre = input_all.max(dim=3)
        indexes_pre = indexes_pre[:, :, :-1]
        mask_pre_num = mask_pre[:, :, :-1].sum().item()
        if mask_pre_num:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[
                mask_pre[:, :, :-1]]).float().sum() / mask_pre_num
        else:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]).float().sum() + 1

        _, indexes_ = target_next.max(2)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_next = input_next.max(2)
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
