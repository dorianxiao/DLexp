[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
# DLexps
国科大深度学习春季课程实验，已更新的内容有
1. Exp1:手写数字识别
2. Exp2:车牌识别
3. Exp3:神经网络语言模型
4. Exp4:情感分类
5. Exp5:猫狗分类

全部代码使用谷歌[Colab](https://colab.research.google.com/)运行，感谢谷歌。

PS：     
#### Colab挂载Google Drive
本人在Google Drive中新建了实验文件夹，结构如下
```
├─mylab
│  ├─Exp1
│  ├─Exp2
  .
  .
  .
│  ├─Exp8
```
现将Colab挂载Google Drive实现Colab操作Google Drive里的文件操作，接下来就可以实现读取里面的数据集了。
```py
# 挂载到Google Drive

from google.colab import drive
drive.mount('/content/gdrive')
```
输入授权码后更改工作目录即可
```py
import os

# mylab 是我在Drive主目录里新建的文件夹
os.chdir(r'/content/gdrive/My Drive/mylab/Exp1')
print(os.getcwd())
```
