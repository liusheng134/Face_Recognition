# ENAS

* 本项目是基于pytorch实现ENAS搜索人脸识别模型架构。

# 数据集

训练集是arcface论文作者提高的清洗过的[ms1m](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)数据集。[测试集](https://pan.baidu.com/s/1PKgRi32PKc3_yNR0ssNmYw)为lfw，cfp-fp，agedb(提取码为：xqjm)。

# 数据增广

```
train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((120, 120), interpolation=3),
        transforms.RandomCrop(112),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
```



## Train
## CNN macro architecture search:

```
cd enas
```

修改train.py里load_data的数据集路径。然后执行以下命令：

```
python train.py
```

## 重零开始训练固定架构CNN

先用搜索阶段训练好的controller采样100个模型，测试这100个模型的准确率，记录准确率最高的模型序列，重零开始训练模型。

执行如下命令。

```
cd ..
python Train.py
```

## Reference 




* https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch
* https://github.com/kcyu2014/eval-nas/tree/6dacf824ebeb7b9554066d65c31bcafa6dd95c28/search_policies/cnn/enas_policy/enas_macro
