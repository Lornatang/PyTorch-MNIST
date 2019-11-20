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
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model.rmdl import RMDL
from utils.eval import accuracy
from utils.misc import AverageMeter

parser = argparse.ArgumentParser(description='PyTorch MNIST Classifier')
parser.add_argument('--train_root', type=str, default="./datasets/mnist_normal/train", help="trainning dataset path.")
parser.add_argument('--valid_root', type=str, default="./datasets/mnist_normal/valid", help="validing dataset path.")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=128, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=28, help='the height / width of the inputs image to network')
parser.add_argument('--lr', type=float, default=0.0001, help="starting lr, every 10 epoch decay 10.")
parser.add_argument('--epochs', type=int, default=50, help="Train loop")
parser.add_argument('--phase', type=str, default='eval', help="train or eval? default:`eval`")
parser.add_argument('--checkpoints_dir', default='./checkpoints', help='folder to output model checkpoints')
parser.add_argument('--model_path', default='./checkpoints/rmdl.pth', help="path to RMDL (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
  os.makedirs("./checkpoints")
except OSError:
  pass

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

# setup gpu driver
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
  CNN = torch.nn.DataParallel(RMDL())
else:
  CNN = RMDL()
if os.path.exists(opt.model_path):
  CNN.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage))

CNN.to(device)


def train():
  try:
    os.makedirs(opt.checkpoints_dir)
  except OSError:
    pass
  ################################################
  #               load train dataset
  ################################################
  dataset = dset.ImageFolder(root=opt.train_root,
                             transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                             ]))

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                           shuffle=True, num_workers=int(opt.workers))

  print(CNN)
  # set train mode
  CNN.train()

  ################################################
  #           Cross Entropy Loss
  ################################################
  criterion = torch.nn.CrossEntropyLoss()

  ################################################
  #            Use Adam optimizer
  ################################################
  optimizer = optim.Adam(CNN.parameters(), lr=opt.lr)

  best_prec1 = 0.
  for epoch in range(opt.epochs):
    # train for one epoch
    print(f"\nBegin Training Epoch {epoch + 1}")
    ################################################
    # Calculate and return the top-k accuracy of the model
    #     so that we can track the learning process.
    ################################################
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    CNN.train()

    end = time.time()
    for i, data in enumerate(dataloader):

      # measure data loading time
      data_time.update(time.time() - end)

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

      if i % 5 == 0:
        print(f"Epoch [{epoch + 1}] [{i}/{len(dataloader)}]\t"
              f"Loss {loss.item():.8f}\t"
              f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
              f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})", end="\r")

    # evaluate on validation set
    print(f"Begin Validation @ Epoch {epoch + 1}")
    prec1 = test()

    # remember best prec@1 and save checkpoint if desired
    best_prec1 = max(prec1, best_prec1)

    print("Epoch Summary: ")
    print(f"\tEpoch Accuracy: {prec1}")
    print(f"\tBest Accuracy: {best_prec1}")

  torch.save(CNN.state_dict(), opt.model_path)


def test():
  ################################################
  #               load valid dataset
  ################################################
  dataset = dset.ImageFolder(root=opt.valid_root,
                             transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                             ]))

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                           shuffle=False, num_workers=int(opt.workers))

  # set eval mode
  CNN.eval()

  # init value
  total = 0.
  correct = 0.
  with torch.no_grad():
    for i, data in enumerate(dataloader):
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

  ################################################
  #               load train dataset
  ################################################
  dataset = dset.ImageFolder(root=opt.valid_root,
                             transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                             ]))

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                           shuffle=False, num_workers=int(opt.workers))

  with torch.no_grad():
    for data in dataloader:
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
    visual()
  else:
    print(opt)
