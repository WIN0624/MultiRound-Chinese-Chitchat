import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class LcccDataset(Dataset):

    def __init__(self, data, tokenizer, n_ctx, mmi=False, max_history=10):
        # mmi模型用回答预测问句，需要将对话逆序
        self.data = data if not mmi else reversed(data)
        self.tokenizer = tokenizer
        self.max_length = n_ctx
        self.mmi = mmi
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        # speaker1对应问者，speaker2对应答者
        self.speaker1, self.speaker2 = self.tokenizer.convert_tokens_to_ids(['[speaker1]', '[speaker2]'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dialog = self.data[index][-2 * self.max_history:]
        return self.process(dialog)

    def process(self, dialog):
        instance = {}
        dialog_ids = [self.tokenizer.cls_token_id]       # mmi不需要将[CLS]逆序
        token_type_ids = [self.tokenizer.cls_token_id]
        for i, sent in enumerate(dialog):
            # extend：一次追加一个序列的所有元素
            sequence = self.tokenizer.encode(sent, add_special_tokens=False)
            dialog_ids.extend(sequence)
            dialog_ids.append(self.tokenizer.sep_token_id)
            # 处理token_type_ids，注意i从0开始
            if not self.mmi:
                token_type_ids.extend([self.speaker2 if i % 2 else self.speaker1] * len(sequence))
            else:
                token_type_ids.extend([self.speaker1 if i % 2 else self.speaker2] * len(sequence))
            token_type_ids.append(self.tokenizer.sep_token_id)

        # 截断超出序列最大长度(n_ctx)的部分
        dialog_ids = dialog_ids[:self.max_length]
        token_type_ids = token_type_ids[:self.max_length]
        instance['input_ids'] = dialog_ids
        instance['token_type_ids'] = token_type_ids
        return instance

    def collate(self, batch):
        """
            - 目标：用于对齐每个dialog张量的长度
            - 以当前batch为输入，进行处理
        """
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=True, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=True, padding_value=self.pad)
        return input_ids, token_type_ids
