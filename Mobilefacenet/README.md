# Face_Recognition_with_Pytorch

本项目基于pytorch训练的Mobilefacenet人脸识别模型，尝试各种方法技巧探索Mobilefacenet模型的上限，并比较各种trick和loss对准确率提高是否有影响。

# 数据集

训练集是arcface论文作者提高的清洗过的[ms1m](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)数据集。为了方便使用和加快训练速度，我转成了lmdb格式（[ms1m_lmdb](https://pan.baidu.com/s/1UwS17OfwBC8kQiBBNFWjVw)，提取码：3fhw）。[测试集](https://pan.baidu.com/s/1PKgRi32PKc3_yNR0ssNmYw)为lfw，cfp-fp，agedb(提取码为：xqjm)。

为了加快训练速度，用DataLoderX替代DataLoder。

关于DataLoderX：[【pytorch】给训练踩踩油门-- Pytorch 加速数据读取](https://blog.csdn.net/shwan_ma/article/details/103331166)

Pytorch多种优化方式集合:https://www.cnblogs.com/king-lps/p/10936374.html

[DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_tutorials.html)也是一个很好的加速方式，但是有些数据预处理，数据增广的方法不支持，代码要自己融入。

# 数据增广

```python
train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Resize((120, 120), interpolation=3),
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
```

# Evaluation 

修改测试集路径和模型路径后：

```
python test_lfw_agedb_cfpfp.py
```

## Train
训练集输入图片尺寸为112x112。

1.用arcface（s=64.0,m=0.50）训练15个epoch，batchsize为200，两张2080ti gpu一起训练，每5000个total_iters评估一次模型在测试集的准确率，并保存模型，损失函数为CrossEntropyLoss，学习率调整策略为CosineAnnealingLR，优化器为SGD：。大概10个epoch后模型准确率停留，上下浮动。

最好的模型准确率为：

LFW average acc:99.3167

CFP_FP average acc:91.4571 

AgeDB30 average acc:94.2667 



2.在1的各项参数基础上加上label smoth训练15个epoch：

```
python Train_mobile.py
```

记录最好结果：

LFW average acc:99.4167

CFP_FP average acc:91.3714

AgeDB30 average acc:95.2000

3.在1的基础上，各项参数保持不变，用arcface（s=64.0,m=0.50）+ focalloss训练至收敛，记录最好结果：

LFW average acc:99.5500

CFP_FP average acc:92.8286

CFP_FP average acc:92.8286

4.借鉴tripletloss减少类内距离，增加类间距离的思想，arcface通过margin来增加类间的距离，没有显式的减少类内距离。centorloss的思想就是减少类间距离。用arcface+centorloss的组合能不能进一步提高模型准确率呢？而且arcface+centorloss的组合不用像triplet那样采样三元组，代码上更简洁。同样用focalloss降低大量简单负样本在训练中所占的权重

执行下面命令训练arcface+centorloss至模型收敛：

```
python Train_arc_focal.py
```

$loss = loss_{focalloss}+\lambda loss_{centorloss}$

$\lambda$值的选取没有用网格搜索等技巧选取最佳值，只简单的选取0.005和0.03两个值。

$\lambda$==0.005:

LFW average acc:99.6333

CFP_FP average acc:93.0000

AgeDB30 average acc:96.1333

[模型地址](https://pan.baidu.com/s/1SeYf64SHpQA6CWFQkopT0w) (提取码：02lf )

$\lambda$==0.03:

LFW average acc:99.7000

CFP_FP average acc:93.5143

AgeDB30 average acc:96.0500

[模型地址](https://pan.baidu.com/s/1l_WE-R-pkpONhI_AXt7Rsw) (提取码：od7f )

5.针对CFP_FP准确率低的问题，做了一组实验，在cutout数据增广里以10%的概率遮挡半张脸训练（简单用遮挡半张对齐好的图片模拟），发现可以提升0.5-0.8%的CFP_FP准确率。更好的方法是用[TP-GAN](https://github.com/HRLTY/TP-GAN)把侧脸图片转成正脸图片，能更显著的提升CFP_FP准确率。

6.项目里有用sknet-101训练的代码，简单用softmax训练了一下发现很容易lfw准确率就达到了99.6%以上。有兴趣的可以用sknet结合上面各种技巧训练，试验效果。

## Reference 
* https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch

  
