# MultiRound-Chinese-Chichat

[TOC]

### Overview

* **Intern Recruitment Test**

  * A chinese chitchat demo based on DialGPT which supports multi-round chichat
  * Time Limit: a week

* **Obstacles**：With no NLP background but basic knowledge on DeepLearning

* **Achievements**

  * Outlined the evolution and structures of NLP models, and dived deep in CDial-GPT and MMI model.

    > RNN/LSTM/GRU -> Seq2Seq -> Seq2Seq with Attention -> Transformer -> GPT2

  * Learned Huggingface for pretrained model fintuning

  * Developed the entire pipeline for this NLP task, including training and inference.

### Demand

* **Target: A multi-round chitchat chabot concerning dailogue history**

    * Inputs: combination of dialogue history and current round input from users
    * Output: candidate response and corresponding loss
    
* **KnowledgeMap**

    <img src="https://cdn.jsdelivr.net/gh/WIN0624/Picgo@main/img/202312160656582.jpg" style="zoom:67%;" />

* **Final Results**

    Though not able to answer with the best candidate sometimes, the chatbot is already capable of generating some history-related candidates.

    <img src="https://cdn.jsdelivr.net/gh/WIN0624/Picgo@main/img/202312160652705.jpg" alt="wecom-temp-56851-e223ff049478c7b9e2f435e5145a7216" width="500" height="300" />

    <img src="https://cdn.jsdelivr.net/gh/WIN0624/Picgo@main/img/202312160658757.png" alt="image-20231216065804695" width="300" height="400" />

### Model

* **Model**

    * **For candidate generate**: CDial-GPT2_LCCC-base

        > 由[CDIAL-GPT](https://github.com/thu-coai/CDial-GPT)提供，利用LCCC-base数据集（微博对话）在NovelGPT基础上进行预训练

    * **For save answer decay: MMI model from DialGPT**

        > The more specific the answers are the more weights they get.

* **Dataset**：[STC数据集](https://cloud.tsinghua.edu.cn/f/372be4a9994b4124810e/?dl=1)

    > STC.json：627M；STC_test.json：2.98M

* **Optimizer**：AdamW

* **WarmUp**：Linear Schedule

* **DecodingStrategy**：temperature + top Sampling

  > Rerank candidate responses based on MMI score. 

### Training Process

* **DatasetSize**: Didn't finish to run the large dataset in this short duration, only 2.98M dataset was applied for fintuning

* **Epochs**: 3 for both CDial-GPT2 and MMI model

* **Accuracy**: CDial-GPT2 50%~60%, MMI model 50%~65%

  > Due to time constraints, most of the time was dedicated to succesffully run the training and inference process, instead of improving the accuracy. 

### Program Design

#### Trianer

**Step1.1 DataLoader**

* **流程**

    <img src="https://cdn.jsdelivr.net/gh/WIN0624/Picgo@main/img/202312160709977.png" alt="image-20231216070915908" width="200" />
    
* **Learning: how to load large dataset**

    ```python
    with open(path, r, encoding='utf-8') as f:
    	dataset = json.loads(f.read())
    ```

**Step1.2 DatasetPreprosessor**

* **pipeline**

   <img src="https://cdn.jsdelivr.net/gh/WIN0624/Picgo@main/img/202312160701252.png" alt="image-20231216070150196" width="340" />
   
* **inputs for CDial-GPT2**

    * **input_ids**: [CLS] question1 [SEP] answer1 [SEP] question2 [SEP] answer2 [SEP]
    * **token_type**: [speaker1] for questions, [speaker2] for answers

**Step2 TrainingLoop**

* **loss calculation**: use token embeddings from output in N-1 position for predict

**Step3 Evaluation**

* 在`model.eval()`模式下，借助测试集对模型进行评估

#### Inference

1. **Get candidate_response by CDial-GPT2**

    * Difficulty: candidate reponses vary in length. need to design batch-process them elegantly.

        <img src="https://cdn.jsdelivr.net/gh/WIN0624/Picgo@main/img/202312160702055.png" alt="image-20231216070230994" style="zoom:40%;" />

2. **Get the most specifc and relevant answer by MMI**

    * **MMI inputs**: concatenate candidate response and dailogue history in reverse order

    * **Output**: the response with minimum loss
    
    * **Action**: output the answer and add it to the history

### ToDo

1. 在已有的checkpoints上，用完整的训练集对两个模型进行训练（增加epoch）
2. 当前只实现了多轮对话，并没有考虑上下文的指代关系。后续可以考虑使用[动态神经网络](https://cs224d.stanford.edu/reports/RaghuvanshiChase.pdf)（传递推理，解决指代关系）

3. [改变编码方式](https://github.com/bojone/nezha_gpt_dialog)
    * 将当前模型的定长编码换成NEZHA的相对位置编码，能接受更长的句子输入
    * UNLM模型：改变mask编码：不预测问句部分，只预测答句部分

## Timeline

> p.s not a full-time project but a project implemented after cources on day

### Theories Learning | 11.21-11.22 

<img src="https://cdn.jsdelivr.net/gh/WIN0624/Picgo@main/img/202312160656368.jpg" alt="wecom-temp-116349-1ce3960b4f9bf055ec9f889f47ca4ab5" style="zoom: 67%;" />

**[Notes]**

* [Knowledge of NLP models](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/1.%E6%A8%A1%E5%9E%8B%E7%9F%A5%E8%AF%86.md)

* [Relevant researches on MultiRound Chatbot](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/3.%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E5%8F%82%E8%80%83%E9%A1%B9%E7%9B%AE.md)
* [How to do BERT Fintuning](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/4.BERT_TUTORIAL.md)
* [How to use HuggingFace transformers](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/5.%20transformers%E4%BD%BF%E7%94%A8.md)
* [Tricks for parameters and metrics chosen for generaton](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/2.%E5%8F%82%E6%95%B0%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BB%B7.md)

### Study on existed research | 11.23 

* **Chinese dialogue generation researches**

    1. [CDIAL-GPT](https://arxiv.org/abs/2008.03946) 

        > github：https://github.com/thu-coai/CDial-GPT

        * 基于中文小说数据预训练12层GPT
        * 提供数据集LCCC，基于LCCC二次预训练GPT
        * 基于STC数据集微调

    2. GPT2 for Chinese chitchat

        > github：https://github.com/yangjianxin1/GPT2-chitchat

        * 预训练模型：Bert tokenizer和GPT-2预训练模型

* **Knowledge on HuggingFace transfomers Application**

    * reference：[BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
    * [notes](https://github.com/WIN0624/MultiRound-Chinese-Chitchat/blob/main/theories/4.BERT_TUTORIAL.md)

### Programming | 11.24-11.26

* Studied source code of GPT2 for Chinese chitchat
* 11.24
    * Training loop
    * Dataset Loader
* 11.25
    * Training loop
    * Customized dataset preprocess, improving process on token2id
* 11.26
    * Inference
