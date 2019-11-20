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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RMDL: Random Multimodel Deep Learning for Classification
* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  RMDL project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
* Link: https://dl.acm.org/citation.cfm?id=3206111
* Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
* Link :  http://www.ijmlc.org/index.php?m=content&c=index&a=show&catid=79&id=823
* Comments and Error: email: kk7nc@virginia.edu
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import random

import torch.nn as nn


class RMDL(nn.Module):

  def __init__(self, nclasses=10,
               min_hidden_layer=1, max_hidden_layer=3,
               min_nodes=128, max_nodes=512,
               dropout=0.05):
    global _Filter
    super(RMDL, self).__init__()
    feature = nn.Sequential()
    values = list(range(min_nodes, max_nodes))
    Layers = list(range(min_hidden_layer, max_hidden_layer))
    Layer = random.choice(Layers)
    Filter = random.choice(values)

    feature.add_module("Conv2D_1", nn.Conv2d(1, Filter, 3, 1, 1))
    feature.add_module("ReLU_1", nn.ReLU(inplace=True))
    feature.add_module("Conv2D_2", nn.Conv2d(Filter, Filter, 3, 1, 1))
    feature.add_module("ReLU_2", nn.ReLU(inplace=True))

    for i in range(0, Layer):
      _Filter = random.choice(values)
      feature.add_module("Conv2D_3", nn.Conv2d(Filter, _Filter, 3, 1, 1))
      feature.add_module("ReLU_3", nn.ReLU(inplace=True))
      feature.add_module("MaxPool2d_1", nn.MaxPool2d((2, 2)))
      feature.add_module("Dropout_1", nn.Dropout(p=dropout))

    classifier = nn.Sequential()
    classifier.add_module("Dense_1", nn.Linear(_Filter * 14 * 14, 256))
    classifier.add_module("ReLU_4", nn.ReLU(inplace=True))
    classifier.add_module("Dropout_2", nn.Dropout(p=dropout))
    classifier.add_module("Dense_2", nn.Linear(256, nclasses))
    classifier.add_module("Softmax", nn.Softmax(dim=1))

    self.feature = feature
    self.classifier = classifier

  def forward(self, x):
    x = self.feature(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
