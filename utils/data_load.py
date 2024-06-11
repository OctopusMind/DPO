import torch
from torch.utils.data import Dataset, DataLoader
import json


class CustomDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.tokenizer = tokenizer
        with open(data_file, 'r', encoding="utf-8") as f:
            self.data = json.load(f)
        self.total_samples = len(self.data)


    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 根据索引加载数据
        # 这里可以根据需要从文件中读取数据并进行预处理
        line = self.data[idx]
        query = line["input"]
        instruction = line["instruction"]
        rejected = line["output"]["rejected"]
        chosen = line["output"]["chosen"]
        messages = [
            {"role": "system", "content": "你是一个非常有帮助和智能的助手。"},
            {"role": "instrution", "content": instruction},
            {"role": "user", "content": query},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer.encode(text=text)
        rejected_inputs = self.tokenizer.encode(text=rejected)
        chosen_inputs = self.tokenizer.encode(text=chosen)
        return [prompt_inputs, rejected_inputs, chosen_inputs]

    def collate_fn(self, batch):
        def statisc_ids(data, masks=None, labels=None, is_pad=False):
            max_length = max([len(i) for i in data])
            attention_masks = []
            return_ids = []
            return_labels = []
            if is_pad:
                for one, mask, label in zip(data, masks, labels):
                    padding_num = max_length - len(one)
                    return_ids.append(one + [self.tokenizer.pad_token_id] * padding_num)
                    return_labels.append(label + [0] * padding_num)
                    attention_masks.append(mask + [0] * padding_num)
            else:
                for one in data:
                    padding_num = max_length - len(one)
                    return_ids.append([self.tokenizer.bos_token_id] * padding_num + one)
                    attention_masks.append([0] * padding_num + [1] * len(one))
            return return_ids, attention_masks, return_labels

        prompt_inputs, prompt_masks, _ = statisc_ids([one[0] for one in batch])
        inputs_ids = []
        inputs_masks = []
        labels = []
        for i in range(1, 3):
            for one, prompt_input, prompt_mask in zip(batch, prompt_inputs, prompt_masks):
                res = one[i]
                inputs_ids.append(prompt_input + res)
                inputs_masks.append(prompt_mask + [1] * len(res))
                labels.append([0] * len(prompt_input) + [1] * len(res))
        inputs_ids, inputs_masks, labels_mask = statisc_ids(inputs_ids, inputs_masks, labels, True)
        return {"inputs_ids": torch.tensor(inputs_ids), "inputs_masks": torch.tensor(inputs_masks),
                "labels_mask": torch.tensor(labels_mask)}


if __name__ == '__main__':
    from config import Config
    from transformers import AutoTokenizer

    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.gpt_model)
    # 创建自定义数据集实例
    dataset = CustomDataset(config.data_path, tokenizer)

    # 创建数据加载器并指定批次大小
    batch_size = 2
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # 使用生成器函数按需读取数据
    for batch in data_loader:
        print()
        # 在每个批次中进行模型训练
        # batch 包含了一个批次的样本数据
        # 在这里执行模型训练操作
        pass
