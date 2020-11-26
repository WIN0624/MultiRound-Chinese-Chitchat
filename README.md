# MultiRound-Chinese-Chichat

A chinese chitchat model based on GPT-2 and DialoGPT which supports multi round chichat.

## Demand

* 实现一个基于对话历史的多轮对话模型

    * 输入是对话历史和用户的当前语句
    * 输出是候选的回答及对应的概率(或是loss)
    * 参考convai、dialoGPT或者使用其他模型如LSTM等
    * 要是更进一步的话可以尝试调研并使用中文数据

* 输出效果图

    <img src="https://gitee.com/WIN0624/document/raw/markdown-picture/img/image-20201121154711357.png" width="60%" height="60%">

## 模型选择

* 预训练模型：Novel-GPT

    > 由[CDIAL-GPT](#)提供，用中文小说数据（5亿词）对12层GPT模型进行预训练

* 微调数据集：由[CDIAL-GPT](#)提供的LCCC数据集
  
    * base：微博对话
    * large：微博对话 + 开源中文对话数据集
    
* 优化器：AdamW
  
* WarmUp：线性增加和衰减
  
* 解码策略：temperature + topp采样 + DialoGPT的MMI模型
  
  > 按照互信息程度，对topp的回答重排，降低泛回答的权重

## 整体框架

### 训练过程

**Step1 载入数据集**

* 流程

    <img src="https://gitee.com/WIN0624/document/raw/markdown-picture/img/image-20201126105423344.png" width="40%" height="40%">
    
* **难点：大数据集的载入**

    ```python
    with open(path, r, encoding='utf-8') as f:
    	dataset = json.loads()
    ```

**Step2.1 模型训练| 数据集处理**

* pipeline

    <img src="https://gitee.com/WIN0624/document/raw/markdown-picture/img/image-20201126105443989.png" width="50%" height="40%">
    
* **对话整合的要点**

    * 将句子拼接成：[CLS] question1 [SEP] answer1 [SEP] question2 [SEP] answer2 [SEP]
    * 对句子进行编码，得到input_ids
    * 令question部分的token_type为[speaker1]，answer部分的token_type为[speaker2]

**Step2.2 模型训练 | TrainingLoop**

* **重点：loss的计算**
    * 用outputs中第n-1个位置的概率分布，预测第n个token

**Step3 模型评估**

* 在`model.eval()`模式下，借助测试集对模型进行评估

### 对话过程

## 改进方向

* [动态神经网络](https://cs224d.stanford.edu/reports/RaghuvanshiChase.pdf)：传递推理，能够解决指代关系
* [编码方式改变](https://github.com/bojone/nezha_gpt_dialog)：将当前模型的定长编码换成NEZHA的相对位置编码，能接受更长的句子输入
* UNLM模型：改变mask编码：不预测问句部分，只预测答句部分

## 推进情况

### 理论知识学习 | 11.21-11.22 

* [模型相关知识](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/1.%E6%A8%A1%E5%9E%8B%E7%9F%A5%E8%AF%86.md)
* [多轮对话参考项目](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/3.%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E5%8F%82%E8%80%83%E9%A1%B9%E7%9B%AE.md)
* [BERT微调项目](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/4.BERT_TUTORIAL.md)
* [HuggingFace transformers使用](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/5.%20transformers%E4%BD%BF%E7%94%A8.md)
* [参数学习的技巧和模型评价指标](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/2.%E5%8F%82%E6%95%B0%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BB%B7.md)

### 已有模型调研 | 11.23 

* **调研当前已有的中文对话生成项目**

    1. [CDIAL-GPT](https://arxiv.org/abs/2008.03946) 

        > github：https://github.com/thu-coai/CDial-GPT

        * 基于中文小说数据预训练12层GPT
        * 提供数据集LCCC，基于LCCC二次预训练GPT
        * 基于STC数据集微调

    2. GPT2 for Chinese chitchat

        > github：https://github.com/yangjianxin1/GPT2-chitchat

        * 预训练模型：Bert tokenizer和GPT-2预训练模型

* **学习HuggingFace transfomers的使用**

    * reference：[BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
    * [笔记](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/4.BERT_TUTORIAL.md)

### 代码实现 | 11.24-11.26

* 研究GPT2 for Chinese chitchat的源码，进行代码复现和优化
* 11.24
    * 源码阅读(训练部分)
    * 实现数据集加载
* 11.25
    * 源码阅读(训练部分)
    * 实现训练过程。
    * 优化数据处理。融合了CDial-GPT的实现逻辑，实现对大数据集的载入，且改进了token2id的方式。
* 11.26
    * 源码阅读(interact部分)