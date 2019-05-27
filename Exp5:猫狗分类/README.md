[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
# Exp5:猫狗分类
+ 数据集来源：[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
> 解压train.zip，共25000张，其中猫狗各占一半。预处理需划分训练集(10000张猫、狗)和验证集(2500张猫、狗)
```
train
├─train
│  ├─dogs
|  └─cats
└─validation
   ├─dogs
   └─cats
```
+ 实验目标：使用TF-slim模块预先构建的InceptionV3进行Fine-tune
## 0. 在Colab中准备数据集
我在Google Drive中 第五个实验目录Exp5里新建data文件夹用于下载数据集。首先在Colab里新建notebook，然后挂载Google Drive：
```py
# 挂载到Google Drive

from google.colab import drive
drive.mount('/content/gdrive')
```
然后更改工作目录
```py
# 更改当前工作目录
import os

os.chdir(r'/content/gdrive/My Drive/mylab/Exp5/data')
print(os.getcwd())
```
接下来步骤是下载Kaggle数据集。首先登陆Kaggle，依次点击 **My Account** -> **Create New API Token**，下载kaggle.json文件。再执行以下操作：
```
!mkdir -p ~/.kaggle

from google.colab import files
files.upload()
```
上传刚刚下载的json文件，继续执行
```
!cp kaggle.json ~/.kaggle/
```
查看dogs相关的数据集
```
!kaggle datasets list -s dogs
```
出现dogs相关的数据集列表
```
ref                                                  title                                           size  lastUpdated          downloadCount  
---------------------------------------------------  ---------------------------------------------  -----  -------------------  -------------  
rahul897/catsdogs                                    cats&dogs                                      216MB  2018-01-05 15:02:32           1546  
jessicali9530/stanford-dogs-dataset                  Stanford Dogs Dataset                          735MB  2019-02-13 05:45:25           3604  
mauricefreund/cats-vs-dogs                           cats_vs_dogs                                     6MB  2017-11-06 13:28:08            219  
mmoreaux/audio-cats-and-dogs                         Audio Cats and Dogs                             49MB  2017-10-05 09:40:26           1859  
chetankv/dogs-cats-images                            Dogs & Cats Images                             216MB  2018-04-19 18:20:08           1432  
nafisur/dogs-vs-cats                                 Dogs_vs_cats                                   217MB  2018-04-25 08:01:49            757  
tongpython/cat-and-dog                               Cat and Dog                                    217MB  2018-04-26 10:56:50           3701  
kmader/dogs-of-zurich                                Dogs of Zurich                                 254KB  2017-03-08 15:07:26            601  
salader/dogs-vs-cats                                 dogs vs cats                                   545MB  2018-11-30 19:31:13            125  
biaiscience/dogs-vs-cats                             Dogs vs Cats                                   814MB  2017-12-05 14:03:33            700  
ppleskov/cute-cats-and-dogs-from-pixabaycom          Cute Cats and Dogs from Pixabay.com              2GB  2019-03-21 04:28:53             61  
aadityanaik/shakespeareworks                         Tarantino Scripts                              508KB  2018-08-14 18:29:50             70  
lucassj/data-for-dogs-and-cats                       data for dogs and cats                         546MB  2018-09-17 23:20:41             45  
chrisj857/cats-vs-dogs-tutorial                      cats_vs_dogs_tutorial                          814MB  2018-02-02 19:12:55            130  
rtatman/5day-data-challenge-signup-survey-responses  5-Day Data Challenge Sign-Up Survey Responses   63KB  2017-12-13 00:09:15            324  
prasunroy/natural-images                             Natural Images                                 171MB  2018-08-11 18:24:11           1543  
siddarthareddyt/cats-and-dogs                        Cats and Dogs                                  816MB  2017-12-27 09:25:45            280  
miljan/stanford-dogs-dataset-traintest               Stanford Dogs Dataset (Train/test)             197MB  2019-02-28 12:22:40             73  
hellokugo/dogs-vs-cats                               dogs vs cats                                   211KB  2018-02-24 03:54:49             39  
codingheerlen/catanddogs                             cat-and-dogs                                   221MB  2019-04-25 13:43:23              6  
```
进行下载
```
!kaggle datasets download biaiscience/dogs-vs-cats --unzip
```
这时data目录下出现两个压缩文件，train.zip和test.zip，这里我们只需解压train.zip
```
!unzip train.zip
```
解压完成后data目录下会生成train目录，目录下也就是我们需要用到的数据，数据准备也就到这里了。

## 1. 实验预处理
这一部分的目的是划分数据集，并生成对应的tfrecord文件。执行[preprocess.py](https://github.com/dorianxiao/DLexp/blob/master/Exp5:%E7%8C%AB%E7%8B%97%E5%88%86%E7%B1%BB/preprocess.py)
```
!python preprocess.py
```
就会在data目录下生成**dogsVScats_train_\*.tfrecord**和**dogsVScats_validation_\*.tfrecord**两个文件。
然后在Exp5目录下导入TF-slim模块
```
!git clone https://github.com/tensorflow/models/
```
导入完成后，进入modles/research/slim/datasets，把[dogsVScats.py](https://github.com/dorianxiao/DLexp/blob/master/Exp5:%E7%8C%AB%E7%8B%97%E5%88%86%E7%B1%BB/dogsVScats.py)和[dataset_factory.py](https://github.com/dorianxiao/DLexp/blob/master/Exp5:%E7%8C%AB%E7%8B%97%E5%88%86%E7%B1%BB/dataset_factory.py)上传进来。这里dogsVScats.py是copy的同目录下flower.py的代码，仅更改了如下地方
```py
_FILE_PATTERN=’dogsVScats_%s_*.tfrecord’
SPLITS_TO_SIZES=(‘train’:20000,’validation’:5000)
_NUM_CLASSES=2
```
dataset_factory.py则在原来代码的基础上注册dogsVScats，新增如下内容
```py
# 导入dogsVScats
from datasets import dogsVScats

# 在datasets_map字典中，增加如下键值
'dogsVScats':dogsVScats,
```
完成之后在slim目录下新建dogsVScats目录，然后在新建的dogsVScats目录下再新建如下四个目录：**data**、**train_dir**、**eval_dir**、**pretrained**。   
+ data:将前面步骤生成的**dogsVScats_train_\*.tfrecord**和**dogsVScats_validation_\*.tfrecord**两个文件copy进来
+ train_dir:保存训练过程中的日志和模型
+ eval_dir:保存验证过程中的日志
+ pretrained：保存Inception V3预训练模型，可点[此处](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)下载并解压，得到inception_v3.ckpt
总结一下在slim模块下需要修改的文件目录如下
```
slim
├─datasets
│  ├─dogsVScats.py        # 新建的文件
|  └─dataset_factory.py   # 修改的文件
├─dogsVScats              # 新建的目录
│  ├─data                 # 新建的目录
|  │  ├─dogsVScats_train_*.tfrecord       # tfrecord文件
|  |  └─dogsVScats_validation_*.tfrecord  # tfrecord文件
│  ├─train_dir            # 新建的目录
│  ├─eval_dir             # 新建的目录
│  └─pretrained           # 新建的目录
|     └─inception_v3.ckpt # 下载的预训练文件
├─train_image_classifier.py
└─eval_image_classifier.py
```
## 2. 训练模型
参数可以按需修改，在Colab中更改运行时，选择GPU。切换到slim目录下[运行](https://github.com/dorianxiao/DLexp/blob/master/Exp5:%E7%8C%AB%E7%8B%97%E5%88%86%E7%B1%BB/cat_dog.ipynb)，接下来是漫长的等待：
```
!python  train_image_classifier.py \
  --train_dir=dogsVScats/train_dir \
  --dataset_name=dogsVScats \
  --dataset_split_name=train \
  --dataset_dir=dogsVScats/data \
  --model_name=inception_v3 \
  --checkpoint_path=dogsVScats/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=10000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=6 \
  --log_every_n_steps=20 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
  ```
## 3. 验证模型
```
!python eval_image_classifier.py \
  --checkpoint_path=dogsVScats/train_dir \
  --eval_dir=dogsVScats/eval_dir \
  --dataset_name=dogsVScats \
  --dataset_split_name=validation \
  --dataset_dir=dogsVScats/data \
  --model_name=inception_v3
```
![实验结果](https://github.com/dorianxiao/DLexp/blob/master/Exp5:%E7%8C%AB%E7%8B%97%E5%88%86%E7%B1%BB/image.png?raw=true)
