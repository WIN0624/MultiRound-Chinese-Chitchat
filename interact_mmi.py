import os
import torch
import datetime
import argparse
from dataset import MyDataset
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel


def set_args():
    parser = argparse.ArgumentParser()
    # Dialog
    parser.add_argument('--response_num', default=5, type=int)
    parser.add_argument("--max_length", default=30, type=int)
    # Decoder
    parser.add_argument('--topk', default=0, type=int)
    parser.add_argument('--topp', default=0.9, type=float)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--repetition_penalty', default=1.5, type=float)
    # Tokenizer and Models
    parser.add_argument('--vocab_path', default='resource/CDial-GPT2_LCCC-base/', type=str)
    parser.add_argument('--dialogue_model', default='models/dialogue/model_epoch3', type=str)
    parser.add_argument('--mmi_model', default='models/mmi/model_epoch3', type=str)
    # Device
    parser.add_argument('--gpu', default='0,1', type=str)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str)
    return parser.parse_args()


def get_next_token(logits, top_k=0, top_p=0.0, filter_values=-float('Inf')):
    # logits: [batch_size, vocab_size]
    # save-check
    assert logits.dim() == 2

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        # topk: 返回(值, 索引)
        # topk的值：[batch_size, topk], 最后一列是每组topk的最小值
        # 若直接取最后一列，会按一维向量返回([batch_size])，需要用None为其增加一维，shape -> [batch_size, min_topk]
        fileter_index = logits < torch.topk(logits, top_k)[0][:, -1, None]
        logits[fileter_index] = filter_values

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        for idx, index in enumerate(sorted_indices_to_remove):
            indices_to_remove = sorted_indices[idx, index]
            logits[idx, indices_to_remove] = filter_values

    # logits: [batch_size, vocab_size]，包含filter_values
    next_token_ids = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return next_token_ids


def main():
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    # Load Dialogue Model
    dialogue_model = GPT2LMHeadModel.from_pretrained(args.dialogue_model)
    dialogue_model.to(args.device)
    dialogue_model.eval()
    # Load MMI Model
    mmi_model = GPT2LMHeadModel.from_pretrained(args.mmi_model)
    mmi_model.to(args.device)
    mmi_model.eval()
    print("")
    print('[{:}] {:>20s}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Start a dialogue! (Use CTRL + C to exit)'))
    history = []

    while True:
        try:
            # 加载对话模型得到多个candidate response
            question = input('user: ')
            """
            # generated_ids记录一次回答中所有response产生过的词
            # finish_states追踪每个response的生成状态
            """
            candidate_responses = []
            generated_ids = set()
            not_finish_states = []
            for _ in range(args.response_num):
                candidate_responses.append([])
                not_finish_states.append(True)

            # 拼接并编码历史对话和当前问句
            history.append(question)
            dialog = MyDataset([history], tokenizer, args.max_length)[0]
            input_ids = torch.tensor(dialog['input_ids']).long().repeat(args.response_num, 1).to(args.device)
            token_type_ids = torch.tensor(dialog['token_type_ids']).long().repeat(args.response_num, 1).to(args.device)
            for _ in range(args.max_length):
                # shape: [not_finish_response_num, length, vocab_size]
                outputs = dialogue_model(input_ids=input_ids, token_type_ids=token_type_ids)
                next_token_logits = outputs[0][:, -1, :]     # shape: [response_num, vocab_size]. torch自动压缩维度

                # 对此前已经生成的词进行一定程度的惩罚
                for token_id in generated_ids:
                    next_token_logits[:, token_id] /= args.repetition_penalty
                # 增大高权重的影响程度
                next_token_logits /= args.temperature
                # 将[UNK]的概率设为无限小
                next_token_logits[:, tokenizer.unk_token_id] = -float('Inf')
                # 获取每个response的下一个token, shape: [batch_size, 1]
                next_token_ids = get_next_token(next_token_logits, args.topk, args.topp)

                # 循环终止判定 和 为下一轮循环更新数据
                # 张量元素逐个与标量比较，将为终止的response保留在input_ids和token_ids中
                not_finish_states = (next_token_ids != tokenizer.sep_token_id).reshape(-1)
                not_finish_num = not_finish_states.sum().item()
                # 对产生的token作记录
                for i, token_id in enumerate(next_token_ids[not_finish_states]):
                    token_id = token_id.item()
                    generated_ids.add(token_id)
                    candidate_responses[i].append(token_id)
                input_ids = torch.cat((input_ids[not_finish_states, :], next_token_ids[not_finish_states]), dim=-1).to(args.device)
                batch_tokens = torch.zeros(not_finish_num, 1) + tokenizer.convert_tokens_to_ids('[speaker2]')
                batch_tokens = batch_tokens.long().to(args.device)
                token_type_ids = torch.cat((token_type_ids[not_finish_states, :], batch_tokens), dim=-1).to(args.device)

                # 若所有response都完成，则not_finish_states不再包含True，满足以下条件，循环终止
                if not_finish_num == 0:
                    for i in range(args.response_num):
                        candidate_responses[i] = "".join(tokenizer.convert_ids_to_tokens(candidate_responses[i]))
                    break

            # 加载mmi模型计算各个candidate response的loss
            # 训练时输入完整对话，最后一句为response，而此时最后一句为问句，处理上稍有不同
            print('candidate response:')
            min_loss = float('Inf')
            best_response = ""
            for response in candidate_responses:
                dialog = MyDataset([history + [response]], tokenizer, args.max_length, mmi=True)[0]
                input_ids = torch.tensor(dialog['input_ids']).long().to(args.device)
                token_type_ids = torch.tensor(dialog['token_type_ids']).long().to(args.device)
                output = mmi_model(
                    input_ids=input_ids[:-1],
                    labels=input_ids[1:],
                    token_type_ids=token_type_ids[:-1]
                )
                loss = output[0].item()
                print("{} loss:{}".format(response, loss))
                if loss < min_loss:
                    min_loss = loss
                    best_response = response
            print('chatbot: ', best_response)
            print("")
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
