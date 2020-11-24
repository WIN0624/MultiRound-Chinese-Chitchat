# MultiRound-Chinese-Chichat

A chinese chitchat model based on GPT-2 and DialoGPT which supports multi round chichat.

## Demand

* 实现一个基于对话历史的多轮对话模型

    * 输入是对话历史和用户的当前语句
    * 输出是候选的回答及对应的概率(或是loss)
    * 参考convai、dialoGPT或者使用其他模型如LSTM等
    * 要是更进一步的话可以尝试调研并使用中文数据

* 输出效果图

    <img src="https://gitee.com/WIN0624/document/raw/markdown-picture/img/image-20201121154711357.png" alt="image-20201121154711357" style="zoom: 33%;" />

## 数据集

* LCCC：https://github.com/thu-coai/CDial-GPT
    * base：微博对话
    * large：微博对话 + 开源中文对话数据集

## 模型选择

* 预训练模型

    * 方案一：Novel-GPT，用中文小说数据（5亿词）对12层GPT模型进行预训练

        > 编码方式：BPE

    * 方案二：BERT tokenizer + GPT-2

        > 编码方式：BERT

## 训练流程

* 预训练模型：中文小说GPT
* 用LCCC-base数据集微调

## 改进方向

* [动态神经网络](#https://cs224d.stanford.edu/reports/RaghuvanshiChase.pdf)：传递推理，能够解决指代关系
* [编码方式改变](#https://github.com/bojone/nezha_gpt_dialog)：将当前模型的定长编码换成NEZHA的相对位置编码，能接受更长的句子输入

## 知识储备



## 推进情况

### 11.21-11.22

理论知识学习

### 11.23

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

