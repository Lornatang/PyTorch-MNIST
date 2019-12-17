# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""SimplerNetV1 in Pytorch."""
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet_v1(nn.Module):
  def __init__(self, nclasses=10):
    super(SimpleNet_v1, self).__init__()

    features = nn.Sequential(
      nn.Conv2d(1, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
      nn.Dropout2d(p=0.1),

      nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
      nn.Dropout2d(p=0.1),

      nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
      nn.Dropout2d(p=0.1),

      nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
      nn.Dropout2d(p=0.1),

      nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
      nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True),

      # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
      nn.Dropout2d(p=0.1),

      nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
      nn.ReLU(inplace=True)
    )

    for m in features.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    self.features = features
    self.classifier = nn.Linear(256, nclasses)

  def forward(self, x):
    x = self.features(x)

    # Global Max Pooling
    x = F.max_pool2d(x, kernel_size=x.size()[2:])
    x = F.dropout2d(x, 0.1)

    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
