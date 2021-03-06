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

import argparse
import os
import random

import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.dataloader
import torchsummary.torchsummary as torchsummary
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model.rmdl import RMDL
from model.simplenet_v1 import SimpleNet_v1
from model.vgg8b import vgg8b
from model.lenet300_100 import LeNet_300_100
from utils.eval import accuracy
from utils.misc import AverageMeter

parser = argparse.ArgumentParser(description='PyTorch MNIST Classifier')
parser.add_argument('--train_root', type=str, default="./datasets/", help="trainning dataset path.")
parser.add_argument('--valid_root', type=str, default="./datasets/", help="validing dataset path.")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=128, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=28, help='the height / width of the inputs image to network')
parser.add_argument('--lr', type=float, default=0.0001, help="starting lr, every 10 epoch decay 10.")
parser.add_argument('--epochs', type=int, default=50, help="Train loop")
parser.add_argument('--phase', type=str, default='eval', help="train or eval? default:`eval`")
parser.add_argument('--checkpoints_dir', default='./checkpoints', help='folder to output model checkpoints')
parser.add_argument('--model', default='rmdl', help="path to Network, default: rmdl. (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--plot', default=True, type=bool, help='Whether to draw the current accuracy of all categories')

opt = parser.parse_args()

try:
  os.makedirs(opt.checkpoints_dir)
except OSError:
  pass

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

# setup gpu driver
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model path
MODEL_PATH = os.path.join(opt.checkpoints_dir, f"{opt.model}.pth")

train_dataset = dset.MNIST(root=opt.train_root,
                           download=True,
                           train=True,
                           transform=transforms.Compose([
                                   transforms.Resize(opt.img_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5,), std=(0.5, )),
                                 ]))
valid_dataset = dset.MNIST(root=opt.valid_root,
                           download=True,
                           train=False,
                           transform=transforms.Compose([
                                   transforms.Resize(opt.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, ), std=(0.5, )),
                                 ]))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=int(opt.workers))

valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size,
                                               shuffle=False, num_workers=int(opt.workers))


################################################
#           Load model struct
################################################
if opt.model == "rmdl":
  CNN = RMDL()
elif opt.model == "simplenet_v1":
  CNN = SimpleNet_v1()
elif opt.model == "vgg8b":
  CNN = vgg8b()
elif opt.model == "lenet":
  CNN = LeNet_300_100()
else:
  CNN = RMDL()

# check gpu numbers.
if torch.cuda.device_count() > 1:
  CNN = torch.nn.DataParallel(CNN())


def train():
  try:
    os.makedirs(opt.checkpoints_dir)
  except OSError:
    pass


  CNN.to(device)
  CNN.train()
  torchsummary.summary(CNN, (1, 28, 28))

  ################################################
  # Set loss function and Adam optimier
  ################################################
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(CNN.parameters(), lr=opt.lr)

  for epoch in range(opt.epochs):
    # train for one epoch
    print(f"\nBegin Training Epoch {epoch + 1}")
    # Calculate and return the top-k accuracy of the model
    # so that we can track the learning process.
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, data in enumerate(train_dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, targets = data
      inputs = inputs.to(device)
      targets = targets.to(device)

      # compute output
      output = CNN(inputs)
      loss = criterion(output, targets)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output, targets, topk=(1, 5))
      losses.update(loss.item(), inputs.size(0))
      top1.update(prec1, inputs.size(0))
      top5.update(prec5, inputs.size(0))

      # compute gradients in a backward pass
      optimizer.zero_grad()
      loss.backward()

      # Call step of optimizer to update model params
      optimizer.step()

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 15 == 0:
        print(f"Epoch [{epoch + 1}] [{i}/{len(train_dataloader)}]\t"
              f"Loss {loss.item():.4f}\t"
              f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
              f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})", end="\r")

    # save model file
    torch.save(CNN.state_dict(), MODEL_PATH)


def test():
  CNN.load_state_dict(torch.load(MODEL_PATH))
  CNN.to(device)
  CNN.eval()

  # init value
  total = 0.
  correct = 0.
  with torch.no_grad():
    for i, data in enumerate(valid_dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, targets = data
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = CNN(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += targets.size(0)
      correct += (predicted == targets).sum().item()

  accuracy = 100 * correct / total
  return accuracy


def visual():
  class_correct = list(0. for _ in range(10))
  class_total = list(0. for _ in range(10))

  CNN.load_state_dict(torch.load(MODEL_PATH))
  CNN.to(device)
  CNN.eval()

  with torch.no_grad():
    for data in valid_dataloader:
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = CNN(inputs)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()

      for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

  classes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
  for i in range(10):
    print(f"Accuracy of {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:.2f}%")


if __name__ == '__main__':
  if opt.phase == "train":
    train()
  elif opt.phase == "eval":
    print("Loading model successful!")
    accuracy = test()
    print(f"\nAccuracy of the network on the test images: {accuracy:.2f}%.\n")
    if opt.plot:
      visual()
  else:
    print(parser.print_help())
