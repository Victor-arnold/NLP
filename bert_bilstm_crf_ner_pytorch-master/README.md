#### BERT-BiLSTM-CRF模型

##### 【简介】使用谷歌的BERT模型在BiLSTM-CRF模型上进行预训练用于中文命名实体识别的pytorch代码

##### 项目结构
```
bert_bilstm_crf_ner_pytorch
    torch_ner
        bert-base-chinese           --- 预训练模型 https://huggingface.co/bert-base-chinese
        data                        --- 放置训练所需数据
        output                      --- 项目输出，包含模型、向量表示、日志信息等
        source                      --- 源代码
            config.py               --- 项目配置，模型参数
            conlleval.py            --- 模型验证
            logger.py               --- 项目日志配置
            models.py               --- bert_bilstm_crf的torch实现
            main.py                 --- 模型训练
            processor.py            --- 数据预处理
            predict.py              --- 模型预测
            utils.py                --- 工具包
```
##### 运行环境
```
torch==1.8.0
pytorch_crf==0.7.2
numpy==1.17.0
transformers==4.9.0
tqdm==4.62.0
PyYAML==5.4.1
tensorboardX==2.4
```

##### 使用方法
- 修改项目配置
- 训练
```
train()
```
- 预测
```
predict()
```
具体可见`train.py`、`predict.py`


