import os
import time
from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig as Config
from network.sst import build_sst
from network.sst_loss import SSTLoss
from pipline.mot_train_dataset import MOTTrainDataset, collate_fn


def adjust_lr(optimizer):
    for param in optimizer.param_groups:
        param['lr'] *= Config.lr_decay


def train():
    # check path
    if not os.path.exists(Config.ckpt_dir):
        os.makedirs(Config.ckpt_dir)

    # init
    writer = SummaryWriter(Config.log_dir)

    # prepare dataset
    print('loading dataset...')
    dataset = MOTTrainDataset()
    dataloader = torch.utils.data.DataLoader(dataset, Config.batch_size, shuffle=True,
                                             num_workers=Config.num_workers, collate_fn=collate_fn, drop_last=True)

    # create model
    net = torch.nn.DataParallel(build_sst())
    if Config.use_cuda:
        net = net.cuda()

    # criterion && optimizer
    criterion = SSTLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=Config.lr_init,
                                momentum=Config.momentum, weight_decay=Config.weight_decay)

    # from training
    start_epoch = 0
    if Config.from_training:
        print('continue trainint from {}'.format(Config.from_training))
        if Config.use_cuda:
            pretrained = torch.load(Config.from_training)
        else:
            pretrained = torch.load(Config.from_training, map_location='cpu')
        net.module.load_state_dict(pretrained['state_dict'])
        optimizer.load_state_dict(pretrained['optimizer'])
        start_epoch = pretrained['epoch'] + 1
    else:
        print('load backbone from {}'.format(Config.backbone))
        if Config.use_cuda:
            backbone = torch.load(Config.backbone)
        else:
            backbone = torch.load(Config.backbone, map_location='cpu')
        net.module.base.layers.load_state_dict(backbone)
    net.train()

    for epoch in range(start_epoch, Config.max_epoch):
        if epoch in Config.lr_epoch:
            adjust_lr(optimizer)

        for index, iter_data in enumerate(tqdm(dataloader)):
            img_pre, img_next, boxes_pre, boxes_next, labels, valid_pre, valid_next = iter_data

            if Config.use_cuda:
                img_pre = img_pre.cuda()
                img_next = img_next.cuda()
                boxes_pre = boxes_pre.cuda()
                boxes_next = boxes_next.cuda()
                with torch.no_grad():
                    valid_pre = valid_pre.cuda()
                    valid_next = valid_next.cuda()
                    labels = labels.cuda()

            # forward
            out = net(img_pre, img_next, boxes_pre, boxes_next)

            loss_pre, loss_next, loss_union, loss_sim, loss, accuracy_pre, accuracy_next, accuracy = \
                criterion(out, labels, valid_pre, valid_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (index + 1) % Config.log_setp == 0:
                print('epoch: {} || iter: {} || lr: {}'.format(epoch, index, optimizer.param_groups[0]['lr']))
                print('loss_pre: {:.4f} || loss_next: {:.4f} || loss_union: {:.4f} || loss_sim: {:.4f} || loss: {:.4f}'
                    .format(loss_pre.item(), loss_next.item(), loss_union.item(), loss_sim.item(), loss.item()))
                log_index = len(dataloader) * epoch + index
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], log_index)
                writer.add_scalar('loss/loss', loss.item(), log_index)
                writer.add_scalar('loss/loss_pre', loss_pre.item(), log_index)
                writer.add_scalar('loss/loss_next', loss_next.item(), log_index)
                writer.add_scalar('loss/loss_sim', loss_sim.item(), log_index)
                writer.add_scalar('accuracy/accuracy', accuracy.item(), log_index)
                writer.add_scalar('accuracy/accuracy_pre', accuracy_pre.item(), log_index)
                writer.add_scalar('accuracy/accuracy_next', accuracy_next.item(), log_index)
            if (index + 1) % Config.save_step == 0:
                ckpt_name = 'sst900_{}_{}.pth'.format(epoch, index)
                ckpt_dict = {'state_dict': net.module.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'epoch': epoch}
                print('saving checkpoint {}'.format(ckpt_name))
                torch.save(ckpt_dict, os.path.join(Config.ckpt_dir, ckpt_name))
    ckpt_dict = {'state_dict': net.module.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(ckpt_dict, os.path.join(Config.ckpt_dir, 'sst900_final.pth'))
    writer.close()


if __name__ == '__main__':
    train()
