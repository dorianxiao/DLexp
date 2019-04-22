[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

数据集的目录划分如下：
```
dataset
├─ train
│    ├─ area
│    ├─ letter
│    └─ province
└─ val
       ├─ area
       ├─ letter
       └─ province
```
数据集数据量不是很大，因此仅将其划分为训练集train和验证集val，均包含province、area和letter三个目录，且每个目录下按照其所包含的字符划分目录存储对应字符的灰度图，大小为20\*20。三个区域包含的字符如下。
```
Province：
("皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新")
Area：("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z")
Letter：
("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z") 
```
