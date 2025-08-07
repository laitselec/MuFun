import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess,load_audio
from ..utils.arguments import DataArguments
from ..utils.constants import *


import transformers
import torch
from torch.utils.data import Dataset



ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        
        image_size = []
        # if 'image' in sources:
        image_files = sources['image'] if 'image' in sources else []
        segs = sources['time'] if 'time' in sources else None
        
        image_list, image_size = load_audio(self.image_preprocess,image_files,segs,self.data_args.image_folder)
        data_dict['image'] = image_list
        data_dict['image_size'] = image_size
        return data_dict
       


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        
        batch['images'] = []
        for instance in instances:
            batch['images'].extend(instance['image'])
        if batch['images']:
            batch['images'] = torch.cat(batch['images'])
        else:
            batch['images'] = None
        batch['image_sizes'] = [instance['image_size'] for instance in instances]
        
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.eval_data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
