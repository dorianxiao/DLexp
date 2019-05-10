# 猫狗分类
+ 数据集来源：[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
> 解压train.zip，共25000张，其中猫狗各占一半。预处理需划分训练集(10000张猫、狗)和验证集(2500张猫、狗)
```
train
├─train
│  ├─dogs
│  ├─cats
├─validation
│  ├─dogs
│  ├─cats

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
```py
!mkdir -p ~/.kaggle

from google.colab import files
files.upload()
```
上传刚刚下载的json文件，继续执行
```py
!cp kaggle.json ~/.kaggle/
```
查看
```py
!kaggle datasets list -s dogs
```
出现dogs相关的数据集
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
