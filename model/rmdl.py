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
* This file is part of  RMDL project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
* Link: https://dl.acm.org/citation.cfm?id=3206111
* Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
* Link :  http://www.ijmlc.org/index.php?m=content&c=index&a=show&catid=79&id=823
* Comments and Error: email: liuchangyu1111@gmail.com
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import random

import torch.nn as nn


class RMDL(nn.Module):

  def __init__(self, nclasses=10,
               min_hidden_layer=3, max_hidden_layer=10,
               min_nodes=128, max_nodes=512,
               dropout=0.05):
    """

    Args:
      nclasses: Integer
        The categories that the image contains.
      min_hidden_layer: Integer
        Lower Bounds of hidden layers of CNN used in RMDL, it will default to 3.
      max_hidden_layer: Integer
        Upper Bounds of hidden layers of CNN used in RMDL, it will default to 10.
      min_nodes: Integer
        Lower bounds of nodes (2D convolution layer) in each layer of CNN used in RMDL, it will default to 128.
      max_nodes: Integer
        Upper bounds of nodes (2D convolution layer) in each layer of CNN used in RMDL, it will default to 512.
      dropout: Float
        between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
    """
    global _Filter
    super(RMDL, self).__init__()
    features = nn.Sequential()
    values = list(range(min_nodes, max_nodes))
    Layers = list(range(min_hidden_layer, max_hidden_layer))
    Layer = random.choice(Layers)
    Filter = random.choice(values)

    features.add_module("Conv2D_1", nn.Conv2d(1, Filter, 3, 1, 1))
    features.add_module("ReLU_1", nn.ReLU(inplace=True))
    features.add_module("Conv2D_2", nn.Conv2d(Filter, Filter, 3, 1, 1))
    features.add_module("ReLU_2", nn.ReLU(inplace=True))

    for i in range(0, Layer):
      _Filter = random.choice(values)
      features.add_module("Conv2D_3", nn.Conv2d(Filter, _Filter, 3, 1, 1))
      features.add_module("ReLU_3", nn.ReLU(inplace=True))
      features.add_module("MaxPool2d_1", nn.MaxPool2d((2, 2)))
      features.add_module("Dropout_1", nn.Dropout(p=dropout))

    classifier = nn.Sequential()
    classifier.add_module("Dense_1", nn.Linear(_Filter * 14 * 14, 256))
    classifier.add_module("ReLU_4", nn.ReLU(inplace=True))
    classifier.add_module("Dropout_2", nn.Dropout(p=dropout))
    classifier.add_module("Dense_2", nn.Linear(256, nclasses))
    classifier.add_module("Softmax", nn.Softmax(dim=1))

    self.feature = features
    self.classifier = classifier

  def forward(self, x):
    x = self.feature(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
