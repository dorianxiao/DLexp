# Exp9:神经机器翻译
1. 在当前目录下
```git clone https://github.com/tensorflow/nmt/```
2. ```mkdir model```
3. 使用记忆力模型进行训练
```
python -m nmt.nmt.nmt \ # 注意包的相对目录
    --attention=scaled_luong \
    --src=zh --tgt=en \
    --vocab_prefix=./data/vocab  \
    --train_prefix=./data/train \
    --dev_prefix=./data/dev  \
    --test_prefix=./data/test \
    --out_dir=./model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
    ```
4. 根据给定文件进行翻译，输出到output_infer
    ```
    python -m nmt.nmt.nmt \ # 注意包的相对目录
    --out_dir=./model \
    --inference_input_file=./infer_file.zh \
    --inference_output_file=./output_infer
    ```
5. 翻译结果![翻译结果](https://raw.githubusercontent.com/dorianxiao/DLexp/master/Exp9%3A%E7%A5%9E%E7%BB%8F%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91/result.png)
