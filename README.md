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

---------------------------------------------------------------------------------

## **SimpleNet: Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures**

Referenced paper : [Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures](https://arxiv.org/abs/1608.06037)

Referenced code : [https://github.com/Coderx7/SimpleNet_Pytorch](https://github.com/Coderx7/SimpleNet_Pytorch)

### Introduction

Major winning Convolutional Neural Networks (CNNs), such as AlexNet, VGGNet, ResNet, GoogleNet, include tens to hundreds of millions of parameters, which impose considerable computation and memory overhead. This limits their practical use for training, optimization and memory efficiency. On the contrary, light-weight architectures, being proposed to address this issue, mainly suffer from low accuracy. These inefficiencies mostly stem from following an ad hoc procedure. We propose a simple architecture, called SimpleNet, based on a set of designing principles, with which we empirically show, a well-crafted yet simple and reasonably deep architecture can perform on par with deeper and more complex architectures. SimpleNet provides a good tradeoff between the computation/memory efficiency and the accuracy. Our simple 13-layer architecture outperforms most of the deeper and complex architectures to date such as VGGNet, ResNet, and GoogleNet on several well-known benchmarks while having 2 to 25 times fewer number of parameters and operations. This makes it very handy for embedded system or system with computational and memory limitations. 

![](https://github.com/Coderx7/SimpleNet/raw/master/SimpNet_V1/images(plots)/SimpleNet_Arch_Larged.jpg)

### Usage

- train
```text
python3 main.py --model simplenet_v1 --phase train
```

- test
```text
python3 main.py --model simplenet_v1
```

### Documentation

Note that we didn’t intend on achieving the state of the art performance here as we are using a single optimization policy without fine-tuning hyper parameters or data-augmentation for a specific task, and still we nearly achieved state-of-the-art on MNIST. **Results achieved using an ensemble or extreme data-augmentation.** 

---------------------------------------------------------------------------------

## **Training Neural Networks with Local Error Signals**

Referenced paper : [Training Neural Networks with Local Error Signals](https://arxiv.org/abs/1901.06656)

Referenced code : [https://github.com/anokland/local-loss](https://github.com/anokland/local-loss)

### Introduction

Supervised training of neural networks for classification is typically performed with a global loss function. The loss function provides a gradient for the output layer, and this gradient is back-propagated to hidden layers to dictate an update direction for the weights. An alternative approach is to train the network with layer-wise loss functions. In this paper we demonstrate, for the first time, that layer-wise training can approach the state-of-the-art on a variety of image datasets. We use single-layer sub-networks and two different supervised loss functions to generate local error signals for the hidden layers, and we show that the combination of these losses help with optimization in the context of local learning. Using local errors could be a step towards more biologically plausible deep learning because the global error does not have to be transported back to hidden layers. A completely backprop free variant outperforms previously reported results among methods aiming for higher biological plausibility.

### Usage

- train
```text
python3 main.py --model vgg8b --phase train
```

- test
```text
python3 main.py --model vgg8b
```

### Documentation

In the tables below, 'pred' indicates a layer-wise cross-entropy loss, 'sim' indicates a layer-wise similarity matching loss, and 'predsim' indicates a 
combination of these losses. For the local losses, the computational graph is detached after each hidden layer.

### Experiments

Results on MNIST with 2 pixel jittering:

| Network         | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---            | :---       | :---        | :---              | :---             | :--                  |
| mlp             | 2.9M       | 0.75        | 0.68              | 0.80             | **0.62**             |
| vgg8b           | 7.3M       | **0.26**    | 0.40              | 0.65             | 0.31                 |
| vgg8b  + cutout | 7.3M       | -           | -                 | -                | 0.26                 |

Results on Fashion-MNIST with 2 pixel jittering and horizontal flipping:

| Network             | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---                | :---       | :---        | :---              | :---             | :--                  |
| mlp                 | 2.9M       | **8.37**    | 8.60              | 9.70             | 8.54                 |
| vgg8b               | 7.3M       | **4.53**    | 5.66              | 5.12             | 4.65                 |
| vgg8b (2x)          | 28.2M      | 4.55        | 5.11              | 4.92             | **4.33**             |
| vgg8b (2x) + cutout | 28.2M      | -           | -                 | -                | 4.14                 |
        
Results on Kuzusjiji-MNIST with no data augmentation:

| Network         | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---            | :---       | :---        | :---              | :---             | :--                  |
| mlp             | 2.9M       | **5.99**    | 7.26              | 9.80             | 7.33                 |
| vgg8b           | 7.3M       | 1.53        | 2.22              | 2.19             | **1.36**             |
| vgg8b + cutout  | 7.3M       | -           | -                 | -                | 0.99                 |

Results on Cifar-10 with data augmentation:

| Network              | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---                 | :--        | :---        | :---              | :---             | :---                 |
| mlp                  | 27.3M      | 33.56       | 32.33             | 33.48            | **30.93**            |
| vgg8b                | 8.9M       | 5.99        | 8.40              | 7.16             | **5.58**             |
| vgg11b               | 11.6M      | 5.56        | 8.39              | 6.70             | **5.30**             |
| vgg11b (2x)          | 42.0M      | 4.91        | 7.30              | 6.66             | **4.42**             |
| vgg11b (3x)          | 91.3M      | 5.02        | 7.37              | 9.34             | **3.97**             |
| vgg11b (3x) + cutout | 91.3M      | -           | -                 | -                | 3.60                 |
        
Results on Cifar-100 with data augmentation:

| Network              | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---                 | :--        | :---        | :---              | :---             | :---                 |
| mlp                  | 27.3M      | 62.57       | 58.87             | 62.46            | **56.88**            |
| vgg8b                | 9.0M       | 26.24       | 29.32             | 32.64            | **24.07**            |
| vgg11b               | 11.7M      | 25.18       | 29.58             | 30.82            | **24.05**            |
| vgg11b (2x)          | 42.1M      | 23.44       | 26.91             | 28.03            | **21.20**            |
| vgg11b (3x)          | 91.4M      | 23.69       | 25.90             | 28.01            | **20.13**            |
        
Results on SVHN with extra training data, but no augmentation:

| Network         | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---            | :--        | :---        | :---              | :---             | :---                 |
| vgg8b           | 8.9M       | 2.29        | 2.12              | 1.89             | **1.74**             |
| vgg8b + cutout  | 8.9M       | -           | -                 | -                | 1.65                 |

Results on STL-10 with no data augmentation:

| Network         | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---            | :---       | :---        | :---              | :---             | :--                  |
| vgg8b           | 11.5M      | 33.08       | 26.83             | 23.15            | **20.51**            |
| vgg8b + cutout  | 11.5M      | -           | -                 | -                | 19.25                |



