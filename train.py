import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import time
import torch
import logging
import datetime
import argparse
from dataset import LcccDataset
from torch.nn import CrossEntropyLoss
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = None


def set_args():
    parser = argparse.ArgumentParser()
    # HyperParameters
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--gradient_accumulation', default=8, type=int)
    # mmi
    parser.add_argument('--train_mmi', default=False)
    # logs
    parser.add_argument('--log_path', default='./', type=str)
    parser.add_argument('--log_step', default=1, type=int)
    # Paths
    parser.add_argument('--vocab_path', default='resource/Novel_GPT/', type=str)
    parser.add_argument('--raw_data_path', default='resource/LCCC-base-split/', type=str)
    parser.add_argument('--pretrained_model', default='resource/Novel_GPT/', type=str)
    parser.add_argument('--token_data_path', default='tokenized_dataset/', type=str)
    parser.add_argument('--mmi_model_output_path', default='models/mmi/', type=str)
    parser.add_argument('--dialogue_model_output_path', default='models/dialogue/', type=str)
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str)
    return parser.parse_args()


def create_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("[%(asctime)s]  %(levelname)-12s | %(message)s", "%Y-%m-%d %H:%M:%S")

    # 创建一个handler，用于写入日志文件
    log_name = f'{args.log_path}Training.log'
    file_handler = logging.FileHandler(filename=log_name, mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_dataset(raw_data_path, tokenizer, n_ctx, saved_path, mmi=False):
    '''
    - 目的
        将对话的多个句子合并成一个输入序列，如：[CLS]裙 子 哪 个 色 呢[SEP]粉 色[SEP]
    - 参数
        - n_ctx：当前模型规定的序列长度，对话合并后的token不能超过该长度
        - train：获取用作训练的数据集
    '''

    dataset = {}
    # for key in ['train', 'valid', 'test']:
    for key in ['train', 'test']:
        data_cache = f'{saved_path}{key}_{type(tokenizer).__name__}'
        if os.path.exists(data_cache):
            dataset[key] = torch.load(data_cache)
            logger.info("there are {} dialogues in {} cached dataset".format(len(dataset[key]), key))
        else:
            # 1. 读入原数据
            with open(f'{raw_data_path}LCCC-base_{key}.json', 'r', encoding='utf-8') as f:
                raw_data = json.loads(f.read())
                logger.info("there are {} dialogues in {} raw dataset".format(len(raw_data), key))
            # 2. 初始化数据集对象
            logger.info(f'Start encoding for {key} dataset...')
            dataset[key] = LcccDataset(raw_data, tokenizer, n_ctx, mmi)
            # 3. 存储数据集
            torch.save(dataset[key], data_cache)
    return dataset


def calculate_loss_and_accuracy(outputs, pad_id, labels, device):
    """
        - 对每个dialog_ids的输出，可以用第n-1个token预测第n个词的概率分布
        - 交叉熵：相当于有vocab_size个分类
        - outputs[0]：[batch, n_ctx, vocab_size]，每一个元素说明下一个token的分类概率分布（每个类出现的可能性）
          labels：[batch, n_ctx]，每一个元素说明当前token的id（实际类别）
    """
    logits = outputs[0]                                         # outputs里面包含了各隐藏层输出
    shifted_logits = logits[..., :-1, :].contiguous()             # 因为交叉熵传入参数时，需要用到view()
    shifted_labels = labels[..., 1:].contiguous().to(device)

    # 将logits压缩成二维，进行交叉熵计算
    # preds: [batch*n_ctx, vocab_size]
    # labels：[batch*n_ctx]
    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)),
                    shifted_labels.view(-1))

    # 统计非pad的词数
    not_pad = shifted_labels.ne(pad_id)         # 非pad的位置索引
    not_pad_num = not_pad.long().sum().item()

    # accurracy
    _, preds = torch.max(shifted_logits, dim=-1)    # max返回最值和对应索引，此处下标即token_id
    correct = (preds == shifted_labels) & not_pad
    correct_num = correct.float().sum().item()

    loss = loss / not_pad_num
    accuracy = correct_num / not_pad_num
    return loss, accuracy


def train(model, train_dataset, args):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        collate_fn=train_dataset.collate
    )

    total_steps = int(len(train_dataset) * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info(f'Strat Training! Total training steps: {str(total_steps)}')

    # Setting optimizer and learning_rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # 记录当前step次数
    current_step = 0
    # 记录运行时间
    total_t0 = time.time()
    # 记录out of memory的次数
    oom_times = 0

    for epoch in range(args.epochs):
        model.train()
        # ===========================
        #         Training
        # ===========================
        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch+1, args.epochs))

        t0 = time.time()

        for idx, batch in enumerate(train_dataloader):
            try:
                # 1. 加载训练样本
                input_ids = batch[0].to(args.device)
                token_type_ids = batch[1].to(args.device)
                # 2. 前向传播，输出:[batch, n_ctx, vocab_size]，
                # 每段对话可以用第n-1个token预测第n个词的概率分布
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
                # 因源码计算交叉熵时没有忽略[PAD]，因此手动计算loss
                pad_id = train_dataset.pad
                loss, accuracy = calculate_loss_and_accuracy(outputs, pad_id, labels=input_ids, device=args.device)

                # 3. 反向传播 + 梯度裁剪
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # 参数更新
                if (idx + 1) % args.gradient_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()       # 在下一轮更新进行前馈之前，清空梯度
                    current_step += 1
                    if current_step % args.log_step == 0:
                        logger.info(
                            "batch {} of epoch {}, loss {}, accuracy {}".format(idx + 1, epoch + 1, loss, accuracy))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_times += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_times))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception
        logger.info('saving model for epoch {}'.format(epoch + 1))
        if args.train_mmi:
            model_path = os.path.join(args.mmi_model_output_path, 'model_epoch{}'.format(epoch + 1))
        else:
            model_path = os.path.join(args.dialogue_model_output_path, 'model_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        # 每个epoch存储一次
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)
        logger.info('epoch {} finished'.format(epoch + 1))
        logger.info('time for one epoch: {}'.format(format_time(time.time() - t0)))

    logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


def format_time(elapsed):
    # 传入总运行秒数，转换为 hh:mm:ss
    # 1. 对秒数四舍五入
    elapsed_rounded = int(round(elapsed))

    # 2. 格式转换
    return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluation(model, test_dataset, args):
    logger.info('Predicting labels for {:,} test sentences'.format(len(test_dataset)))
    model.eval()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(test_dataset),
        collate_fn=test_dataset.collate
    )
    for idx, batch in enumerate(test_dataloader):
        input_ids = batch[0].to(args.device)
        token_type_ids = batch[1].to(args.device)

        with torch.no_grad():
            # 用元组形式(loss, logits)时，才输入labels
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
            pad_id = test_dataset.pad
            loss, accuracy = calculate_loss_and_accuracy(outputs, pad_id, labels=input_ids, device=args.device)
            logger.info("batch {} of {}, loss {}, accuracy {}".format(idx + 1, len(test_dataset), loss, accuracy))
    logger.info('     DONE.')


def main():
    args = set_args()
    global logger
    logger = create_logger(args)
    logger.info('using device:{}'.format(args.device))
    logger.info('Initilizing tokenizer ...')
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    vocab_size = len(tokenizer)

    logger.info('Loading pretrained model ...')
    model = GPT2LMHeadModel.from_pretrained(args.vocab_path)
    n_ctx = model.config.to_dict().get('n_ctx')     # 获取模型规定的序列长度
    model.resize_token_embeddings(vocab_size)       # 修改预训练模型的vocab大小
    model.to(args.device)

    logger.info('Loading data for training and evaluation ...')
    dataset = get_dataset(args.raw_data_path, tokenizer, n_ctx, args.token_data_path, args.train_mmi)
    train(model, dataset['train'], args)
    evaluation(model, dataset['test'], args)


if __name__ == '__main__':
    main()
