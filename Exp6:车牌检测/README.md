[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
# Exp6:车牌检测
代码结构
```
Exp6
├─datasets
│  ├─train
│  └─val
├─mrcnn
│  ├─__init__.py
│  ├─config.py
│  ├─model.py
│  ├─parallel_model.py
│  ├─utils.py
│  └─visualize.py
└─LPD.py
```
[inspect_train_LPD_data.ipynb](https://github.com/dorianxiao/DLexp/blob/master/Exp6:%E8%BD%A6%E7%89%8C%E6%A3%80%E6%B5%8B/inspect_train_LPD_data.ipynb)进行数据读取和训练，[test_LPD.ipynb](https://github.com/dorianxiao/DLexp/blob/master/Exp6:%E8%BD%A6%E7%89%8C%E6%A3%80%E6%B5%8B/test_LPD.ipynb)进行预测，结果如下![](https://raw.githubusercontent.com/dorianxiao/DLexp/master/Exp6%3A%E8%BD%A6%E7%89%8C%E6%A3%80%E6%B5%8B/result.png)
