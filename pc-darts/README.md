# PC-DARTS-face

本项目是基于pytorch的用pc-darts搜索人脸识别模型架构。

[PC-DARTS论文地址](https://openreview.net/forum?id=BJlS634tPr)

# 数据集

训练集是arcface论文作者提高的清洗过的[ms1m](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)数据集。[测试集](https://pan.baidu.com/s/1PKgRi32PKc3_yNR0ssNmYw)为lfw，cfp-fp，agedb(提取码为：xqjm)。

为了加快训练速度，用DataLoderX替代DataLoder。

关于DataLoderX：[【pytorch】给训练踩踩油门-- Pytorch 加速数据读取](https://blog.csdn.net/shwan_ma/article/details/103331166)

Pytorch多种优化方式集合:https://www.cnblogs.com/king-lps/p/10936374.html

[DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_tutorials.html)也是一个很好的加速方式，但是有些数据预处理，数据增广的方法不支持，代码要自己融入。

face_search/load_dataset.py文件里有各自加快数据读取的函数，包括lmdb，DALI，DALI+LMDB，DALI+MXNET数据格式。

# 训练

pytorch DataParallel训练会造成显存使用不平衡的问题，使用[BalancedDataParallel](http://aiuai.cn/aifarm1328.html)可减缓这周情况。

搜索模型架构：

```
cd face_search
python train_search_ms1m.py
```

训练的log文件里会输出模型架构序列和测试集准确率。

选取最优架构保存在genotypes.py，

## Reference 
* https://github.com/yuhuixu1993/PC-DARTS
