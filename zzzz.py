'''
@Author: rayenwang
@Date: 2019-07-18 10:38:45
@LastEditTime: 2019-07-23 20:56:43
@Description: 
'''

import torch
import torch.nn
import torchvision.models

# model = torchvision.models.resnet50()
# x = torch.rand(1, 3, 512, 512)
# for m in list(model.children())[:-1]:
#     x = m(x)
#     print(x.shape)
# x = model.fc(x.squeeze(2).squeeze(2))
# print(x.shape)
# torch.save(model.state_dict(),'test.pth')