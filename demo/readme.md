# 感觉实现一个完整的框架坑挖大了，先实现一个简单的MNIST的识别吧

和传统的类似于pytorch这种训练框架最大的是，这里的反向传播的过程是直接给出实现的，可以通过的调用backward来求得导数，
但是pytorch或者TensorFlow这种都是基于动态计算图的方式，来实现反向传播的。

## layer
layer是对常用算子的抽象，包含了MNSIT中用到的 Flatten，Sigmoid，ReLu，Softmax，CrossEntropy，Linear这几个算子，
实现中包含各个算子的前向传播和反向传播的过程。

## net
net中是对网络模型的抽象，包含了常见的构建网络，网络的前向传播和反向传播以及梯度下降这些常见的概念。

## train_mnist

是手写数字集的训练代码。（训练前请解压数据集）

```shell
python train_mnist.py
```
即可完成对手写数字集的训练。