# PyTorch-MNIST

The project library contains some of the best performing models. 
But this is only shown on the MNIST handwritten data set, and the effects on other data sets may not be as good. 
The code is implemented using the PyTorch open source library.

---------------------------------------------------------------------------------

## **RMDL: Random Multimodel Deep Learning for Classification**

Referenced paper : [RMDL: Random Multimodel Deep Learning for Classification](https://arxiv.org/abs/1805.01890)

Referenced paper : [An Improvement of Data Classification Using Random Multimodel Deep Learning (RMDL)](https://arxiv.org/abs/1808.08121)

### Introduction

A new ensemble, deep learning approach for classification. 
Deep learning models have achieved state-of-the-art results across many domains. 
RMDL solves the problem of finding the best deep learning structure and architecture while simultaneously improving robustness and accuracy through ensembles of deep learning architectures. 
RDML can accept as input a variety data to include text, video, images, and symbolic.

![](https://camo.githubusercontent.com/9c65085618b4e97694a05b9dce9db3771679bd34/687474703a2f2f6b6f77736172692e6e65742f6f6e657765626d656469612f524d444c2e6a7067)

Random Multimodel Deep Learning (RDML) architecture for classification. RMDL includes 3 Random models, oneDNN classifier at left, one Deep CNN classifier at middle, and one Deep RNN classifier at right (each unit could be LSTMor GRU).

### Usage

- train
```text
python3 main.py --model rmdl --phase train
```

- test
```text
python3 main.py --model rmdl
```

### Documentation

The exponential growth in the number of complex datasets every year requires more enhancement in machine learning methods to provide robust and accurate data classification. Lately, deep learning approaches have been achieved surpassing results in comparison to previous machine learning algorithms on tasks such as image classification, natural language processing, face recognition, and etc. The success of these deep learning algorithms relys on their capacity to model complex and non-linear relationships within data. However, finding the suitable structure for these models has been a challenge for researchers. This paper introduces Random Multimodel Deep Learning (RMDL): a new ensemble, deep learning approach for classification. RMDL solves the problem of finding the best deep learning structure and architecture while simultaneously improving robustness and accuracy through ensembles of deep learning architectures. In short, RMDL trains multiple models of Deep Neural Network (DNN), Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) in parallel and combines their results to produce better result of any of those models individually. To create these models, each deep learning model has been constructed in a random fashion regarding the number of layers and nodes in their neural network structure. The resulting RDML model can be used for various domains such as text, video, images, and symbolic. In this Project, we describe RMDL model in depth and show the results for image and text classification as well as face recognition. For image classification, we compared our model with some of the available baselines using MNIST and CIFAR-10 datasets. Similarly, we used four datasets namely, WOS, Reuters, IMDB, and 20newsgroup and compared our results with available baselines. Web of Science (WOS) has been collected by authors and consists of three sets~(small, medium and large set). Lastly, we used ORL dataset to compare the performance of our approach with other face recognition methods. These test results show that RDML model consistently outperform standard methods over a broad range of data types and classification problems.

## **Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures**

Referenced paper : [Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures](https://arxiv.org/abs/1608.06037)

Referenced paper : [https://github.com/Coderx7/SimpleNet_Pytorch](https://github.com/Coderx7/SimpleNet_Pytorch)

### Introduction

Major winning Convolutional Neural Networks (CNNs), such as AlexNet, VGGNet, ResNet, GoogleNet, include tens to hundreds of millions of parameters, which impose considerable computation and memory overhead. This limits their practical use for training, optimization and memory efficiency. On the contrary, light-weight architectures, being proposed to address this issue, mainly suffer from low accuracy. These inefficiencies mostly stem from following an ad hoc procedure. We propose a simple architecture, called SimpleNet, based on a set of designing principles, with which we empirically show, a well-crafted yet simple and reasonably deep architecture can perform on par with deeper and more complex architectures. SimpleNet provides a good tradeoff between the computation/memory efficiency and the accuracy. Our simple 13-layer architecture outperforms most of the deeper and complex architectures to date such as VGGNet, ResNet, and GoogleNet on several well-known benchmarks while having 2 to 25 times fewer number of parameters and operations. This makes it very handy for embedded system or system with computational and memory limitations. 

![](https://github.com/Coderx7/SimpleNet/raw/master/SimpNet_V1/images(plots)/SimpleNet_Arch_Larged.jpg)

### Usage

- train
```text
python3 main.py --model rmdl --phase train
```

- test
```text
python3 main.py --model rmdl
```

### Documentation

Note that we didn’t intend on achieving the state of the art performance here as we are using a single optimization policy without fine-tuning hyper parameters or data-augmentation for a specific task, and still we nearly achieved state-of-the-art on MNIST. **Results achieved using an ensemble or extreme data-augmentation.** 

---------------------------------------------------------------------------------