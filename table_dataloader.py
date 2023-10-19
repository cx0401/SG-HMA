import json
import os
import pickle
import random
import time
import warnings
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from wsgiref.validate import InputWrapper

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from filelock import FileLock

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import copy
import random


logger = logging.get_logger(__name__)


DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this input_ids script for pointers: {0}"
)


class Data2TextTrainDataset(Dataset):

    def __init__(self, 
        tokenizer: PreTrainedTokenizer,
        src_path: str, 
        tgt_path: str, 
        block_size: int, 
        bos_tok:str, 
        eos_tok:str, 
        metric="BLEU",
        rewrite="no",
        table="no",
        mask="no",
        sort="yes",
        model_type="gpt2",
        dataset="e2e",
        **kwargs):

        assert os.path.isfile(src_path), f"Input file path {src_path} not found"
        assert os.path.isfile(tgt_path), f"Input file path {tgt_path} not found"
        print(f"Creating features from dataset file at {src_path} and {tgt_path}")

        self.tok = tokenizer
        self.block_size =  block_size
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.table = table
        self.rewrite = rewrite
        self.mask = mask
        self.model_type = model_type
        self.dataset = dataset

        # 获取文本数据
        src_origin = [i.replace("\n", "") for i in open(src_path, "r")]


        ref_lines = [[]]

        for tgt in open(tgt_path, "r"):
            if tgt == "\n":
                ref_lines.append([])
            else:
                ref_lines[-1].append("{}".format(tgt.replace("\n", "").strip()))
        ref_lines.pop()


        self.tgt_lines = [ref for refs in ref_lines for ref in refs]
        self.src_lines = [src for i, src in enumerate(src_origin) for _ in ref_lines[i]]
        if table == "yes":
            table_origin = json.load(open(kwargs["table_path"], "r"))
            self.tables = [table for i, table in enumerate(table_origin) for _ in ref_lines[i]]
        
        # 处理负样本
        if rewrite == "yes":
            assert "cand_path" in kwargs and os.path.exists(kwargs["cand_path"])
            src_cand_lines = [[]]
            for cand in open(kwargs["cand_path"], "r"):
                if cand == "\n":
                    src_cand_lines.append([])
                else:
                    src_cand_lines[-1].append("{} {}".format(cand.replace("\n", "").strip(), eos_tok))
            src_cand_lines.pop()
            if sort == "yes":
                assert "cand_metrics_path" in kwargs and os.path.exists(kwargs["cand_metrics_path"])
                src_cand_metrics = json.load(open(kwargs["cand_metrics_path"], "r"))[metric]
                src_cand_lines = [[j for _, j in sorted(zip(src_cand_metrics[i], src_cand_lines[i]), reverse=True)] for i, _ in enumerate(src_cand_lines)]

            self.src_cand_lines  = [cand for i, cand in enumerate(src_cand_lines) for _ in ref_lines[i]]

            assert len(self.src_cand_lines) == len(self.src_lines) == len(self.tgt_lines)
        return


    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        if self.model_type in ["t5", "bart"]:
            input_ids = self.tok(self.src_lines[idx], add_special_tokens=True, is_split_into_words=False, max_length=self.block_size)["input_ids"] 
            labels = self.tok(self.tgt_lines[idx], add_special_tokens=True, is_split_into_words=False, max_length=self.block_size)["input_ids"]
            results = {"input_ids": torch.tensor(input_ids, dtype=torch.int64),"labels": torch.tensor(labels, dtype=torch.int64)}
            if self.rewrite == "yes":
                cand_input_lines = [self.tgt_lines[idx]] + self.src_cand_lines[idx]
                cand_input_ids = self.tok(cand_input_lines, add_special_tokens=True, is_split_into_words=False, max_length=self.block_size, padding=True)["input_ids"]
                cand_masks = copy.deepcopy(cand_input_ids)
                for i, _ in enumerate(cand_input_ids):
                    cand_masks[i] = [True] * len(cand_masks[i])
                results.update({"cand_input_ids" : torch.tensor(cand_input_ids, dtype=torch.int64), "cand_masks" : torch.tensor(cand_masks, dtype=torch.bool)})

        
        if self.model_type == "gpt2":
            edited_sent = f"{self.src_lines[idx]} {self.bos_tok} {self.tgt_lines[idx]} {self.eos_tok}"
            input_ids = self.tok(edited_sent, add_special_tokens=True, is_split_into_words=False)["input_ids"]
            sep_idx = input_ids.index(self.tok.bos_token_id)
            input_ids = input_ids[:min(self.block_size - len(input_ids) + sep_idx, sep_idx)] + input_ids[sep_idx:]
            
            labels = copy.deepcopy(input_ids)
            sep_idx = labels.index(self.tok.bos_token_id)
            if self.dataset != "numericNLG":
                labels[:sep_idx] = [-100] * sep_idx

            results = {"input_ids": torch.tensor(input_ids, dtype=torch.int64),"labels": torch.tensor(labels, dtype=torch.int64)}

            if self.rewrite == "yes":
                cand_input_lines = [self.src_lines[idx] + " " + self.bos_tok + j for j in [self.tgt_lines[idx]] + self.src_cand_lines[idx]]
                cand_input_ids = self.tok(cand_input_lines, add_special_tokens=True, is_split_into_words=False, padding=True,  max_length=self.block_size)["input_ids"]
                sep_idx = cand_input_ids[0].index(self.tok.bos_token_id)
                cand_input_ids = [cand[:min(self.block_size - len(cand) + sep_idx, sep_idx)] + cand[sep_idx:] for cand in cand_input_ids]

                cand_masks = copy.deepcopy(cand_input_ids)
                for i, elem in enumerate(cand_input_ids):
                    sep_idx = elem.index(self.tok.bos_token_id)
                    cand_masks[i] = [True] * len(cand_masks[i])
                    cand_masks[i][:sep_idx] = [False] * sep_idx
                results.update({"cand_input_ids" : torch.tensor(cand_input_ids, dtype=torch.int64), "cand_masks" : torch.tensor(cand_masks, dtype=torch.bool)})
            else:
                cand_input_ids = None
                cand_masks = None
        
        if self.table == "yes":            
            caption_ids = self.tok.batch_encode_plus([self.tables[idx]['caption']], max_length=self.block_size, return_tensors="pt", pad_to_max_length=False, truncation=True)
            caption_ids = caption_ids["input_ids"].squeeze(0)

            rows_ids = [[self.tok(j, return_tensors="pt")['input_ids'] for j in  i] for i in self.tables[idx]['row_subtree']]
            columns_ids = [[self.tok(j, return_tensors="pt")['input_ids'] for j in  i] for i in self.tables[idx]['column_subtree']]
            content_ids = [[torch.cat([self.tok(j[v], return_tensors="pt")['input_ids'] for v in j], dim=-1) for j in self.tables[idx]['mp'][i]] for i in self.tables[idx]['mp']]

            rows_ids_flat = [j for i in rows_ids for j in i]
            columns_ids_flat = [j for i in columns_ids for j in i]

            ## row与column对caption准备位置掩码
            rows_caption_pos = torch.cat([torch.cat([torch.tensor([h] * j.shape[-1]) for num, j in  enumerate(i)]) for h, i in enumerate(rows_ids)])
            columns_caption_pos = torch.cat([torch.cat([torch.tensor([h] * j.shape[-1]) for num, j in  enumerate(i)]) for h, i in enumerate(columns_ids)])
            
            ## content掩码准备位置掩码            
            rows_pos = [torch.cat([torch.tensor([num] * j.shape[-1]) for num, j in  enumerate(i)]) for i in rows_ids]
            columns_pos = [torch.cat([torch.tensor([num] * j.shape[-1]) for num, j in  enumerate(i)]) for i in columns_ids]

            rows_ids = [torch.cat(i, dim=-1) for i in rows_ids]
            columns_ids = [torch.cat(i, dim=-1) for i in columns_ids]

            ## row和column掩码准备位置掩码
            rows_flat_pos = [num for num, j in  enumerate(rows_ids_flat) for k in range(j.shape[-1])]
            columns_flat_pos = [num for num, j in  enumerate(columns_ids_flat) for k in range(j.shape[-1])]

            #记录孩子节点编号
            id_node_row = {}
            count_num = 0
            for i, level in enumerate(self.tables[idx]['row_subtree']):
                for j, node in enumerate(level):
                    id_node_row[node + "_" + str(i)] = count_num
                    count_num += 1
            parents_row = {i:[i] for i in range(len(id_node_row))} 
            for i, level in enumerate(self.tables[idx]['row_subtree']):
                for j, node in enumerate(level):
                    for k, child in enumerate(level[node]):
                        parents_row[id_node_row[[k for k in child][0]+ "_" + str(i + 1)]].append(id_node_row[node + "_" + str(i)])
            
            #记录孩子节点编号
            id_node_column = {}
            count_num = 0
            for i, level in enumerate(self.tables[idx]['column_subtree']):
                for j, node in enumerate(level):
                    id_node_column[node+ "_" + str(i)] = count_num
                    count_num += 1
            parents_column = {i:[i] for i in range(len(id_node_column))} 
            for i, level in enumerate(self.tables[idx]['column_subtree']):
                for j, node in enumerate(level):
                    for k, child in enumerate(level[node]):
                        parents_column[id_node_column[[k for k in child][0]+ "_" + str(i + 1)]].append(id_node_column[node+ "_" + str(i)])


            if self.mask == "yes":
                # content掩码实现
                # contents_rows_mask = [[torch.where(rows_pos[-1] != i, torch.ones([rows_pos[-1].shape[-1]], dtype=torch.bool), torch.zeros([(rows_pos[-1].shape[-1])], dtype=torch.bool)).unsqueeze(0).repeat(val_j.shape[-1], 1) for j, val_j in enumerate(val_i)] for i, val_i in enumerate(content_ids)]
                contents_rows_mask = [[torch.where(rows_pos[-1] != i, torch.ones([rows_pos[-1].shape[-1]]), torch.zeros([(rows_pos[-1].shape[-1])])).unsqueeze(0).repeat(val_j.shape[-1], 1) == 1 for j, val_j in enumerate(val_i)] for i, val_i in enumerate(content_ids)]
                contents_columns_mask = [[torch.where(columns_pos[-1] != j, torch.ones([columns_pos[-1].shape[-1]]), torch.zeros([columns_pos[-1].shape[-1]])).unsqueeze(0).repeat(val_j.shape[-1], 1)==1 for j, val_j in enumerate(val_i)] for i, val_i in enumerate(content_ids)]
                
                content_ids = torch.cat([torch.cat(i, dim=-1) for i in content_ids], dim=-1).squeeze(0)
                contents_rows_mask = torch.cat([torch.cat(i, dim=0) for i in contents_rows_mask], dim=0)
                contents_columns_mask = torch.cat([torch.cat(i, dim=0) for i in contents_columns_mask], dim=0)

                contents_rows_mask = torch.cat((torch.zeros(contents_rows_mask.shape[0], sum([i.shape[-1] for i in rows_ids[:-1]]), dtype=torch.bool), contents_rows_mask), dim=-1)
                contents_columns_mask = torch.cat((torch.zeros(contents_columns_mask.shape[0], sum([i.shape[-1] for i in columns_ids[:-1]]), dtype=torch.bool), contents_columns_mask), dim=-1)

                # row和column掩码实现
                rows_self_mask = torch.tensor([[0 if j in parents_row[i] else 1 for j in rows_flat_pos] for i in rows_flat_pos], dtype=torch.bool)
                columns_self_mask = torch.tensor([[0 if j in parents_column[i] else 1 for j in columns_flat_pos] for i in columns_flat_pos], dtype=torch.bool)

                # row和column对caption掩码实现
                rows_caption_mask = torch.where(rows_caption_pos.unsqueeze(0).repeat(caption_ids.shape[-1], 1).T==0, torch.zeros(rows_caption_pos.shape[-1], caption_ids.shape[-1]), torch.ones([rows_caption_pos.shape[-1], caption_ids.shape[-1]])) == 1
                columns_caption_mask = torch.where(columns_caption_pos.unsqueeze(0).repeat(caption_ids.shape[-1], 1).T==0, torch.zeros(columns_caption_pos.shape[-1], caption_ids.shape[-1]), torch.ones([columns_caption_pos.shape[-1], caption_ids.shape[-1]])) == 1

                results.update({
                    "caption_ids": caption_ids,
                    "content_ids": content_ids,
                    "contents_rows_mask": contents_rows_mask,
                    "contents_columns_mask": contents_columns_mask,
                    "rows_self_mask": rows_self_mask,
                    "columns_self_mask": columns_self_mask,
                    "rows_caption_mask": rows_caption_mask,
                    "columns_caption_mask": columns_caption_mask
                })

            else:
                content_ids = torch.cat([torch.cat(i, dim=-1) for i in content_ids], dim=-1).squeeze(0)
                results.update({
                    "content_ids": content_ids,
                })
            
            rows_ids = torch.cat(rows_ids, dim=-1).squeeze(0)
            columns_ids = torch.cat(columns_ids, dim=-1).squeeze(0)
            results.update({
                    "rows_ids": rows_ids,
                    "columns_ids": columns_ids
            })

            if rows_ids.dtype != torch.int64:
                print("error")

        return results


@dataclass
class DataCollatorForData2TextTrain:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    format_mode: str = 'cat'
    mlm_probability: float = 0.15

    def __call__(
        self, examples) -> Dict[str, torch.Tensor]:
        padding_values = {
            "label": -100,
            "id": self.tokenizer.pad_token_id,
            "mask": False
        }
        batch = {key:[e[key] for e in examples] for key in examples[0].keys()}
        match = {key: [pad for pad in padding_values if pad in key][0] for key in batch}
        batch = {key: self._tensorize_batch(batch[key], padding_values[match[key]]) for key in batch}
        return batch

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]], padding_value
    ) -> torch.Tensor:

        if examples[0].dtype != torch.int64 and examples[0].dtype != torch.bool:
            print("error")
        
        if len(examples[0].shape) == 1:
            return pad_sequence(examples, batch_first=True, padding_value=padding_value)
        elif len(examples[0].shape) == 2:
            if examples[0].type() == "torch.BoolTensor":
                max_len = max([e.shape[0] for e in examples])
                examples = [torch.cat((e, torch.full((max_len - e.shape[0], e.shape[1]), padding_value).to(e.device)), dim=0) for e in examples]
            max_len = max([e.shape[1] for e in examples])
            examples = torch.cat([torch.cat((e, torch.full((e.shape[0] ,max_len - e.shape[1]), padding_value).to(e.device)), dim=-1).unsqueeze(0) for e in examples], dim=0)
            return examples
        assert False

        
class Data2TextGenerationDataset(Dataset):
    def __init__(self, 
        tokenizer: PreTrainedTokenizer,
        src_path: str, 
        block_size: int, 
        bos_tok:str, 
        eos_tok:str, 
        table="no",
        mask="no",
        model_type="gpt2",
        **kwargs):

        assert os.path.isfile(src_path), f"Input file path {src_path} not found"

        print(f"Creating features from dataset file at {src_path}")

        self.tok = tokenizer
        self.block_size =  block_size
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.table = table
        self.mask = mask
        self.model_type = model_type

        # 获取文本数据
        self.src_lines = open(src_path, "r").readlines()
        if table == "yes":
            self.tables = json.load(open(kwargs["table_path"], "r"))

        return


    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        input_line = self.src_lines[idx]
        input_ids = self.tok(input_line, add_special_tokens=True, truncation=True, max_length=self.block_size - 128, is_split_into_words=False)["input_ids"]
        if self.model_type == "gpt2":
            input_ids += [self.tok.bos_token_id]
        results = {"input_ids": torch.tensor(input_ids, dtype=torch.int64)}

        if self.table == "yes":            
            caption_ids = self.tok.batch_encode_plus([self.tables[idx]['caption']], max_length=self.block_size, return_tensors="pt", pad_to_max_length=False, truncation=True)
            caption_ids = caption_ids["input_ids"].squeeze(0)

            rows_ids = [[self.tok(j, return_tensors="pt")['input_ids'] for j in  i] for i in self.tables[idx]['row_subtree']]
            columns_ids = [[self.tok(j, return_tensors="pt")['input_ids'] for j in  i] for i in self.tables[idx]['column_subtree']]
            content_ids = [[torch.cat([self.tok(j[v], return_tensors="pt")['input_ids'] for v in j], dim=-1) for j in self.tables[idx]['mp'][i]] for i in self.tables[idx]['mp']]

            rows_ids_flat = [j for i in rows_ids for j in i]
            columns_ids_flat = [j for i in columns_ids for j in i]

            ## row与column对caption准备位置掩码
            rows_caption_pos = torch.cat([torch.cat([torch.tensor([h] * j.shape[-1]) for num, j in  enumerate(i)]) for h, i in enumerate(rows_ids)])
            columns_caption_pos = torch.cat([torch.cat([torch.tensor([h] * j.shape[-1]) for num, j in  enumerate(i)]) for h, i in enumerate(columns_ids)])
            
            ## content掩码准备位置掩码            
            rows_pos = [torch.cat([torch.tensor([num] * j.shape[-1]) for num, j in  enumerate(i)]) for i in rows_ids]
            columns_pos = [torch.cat([torch.tensor([num] * j.shape[-1]) for num, j in  enumerate(i)]) for i in columns_ids]

            rows_ids = [torch.cat(i, dim=-1) for i in rows_ids]
            columns_ids = [torch.cat(i, dim=-1) for i in columns_ids]

            ## row和column掩码准备位置掩码
            rows_flat_pos = [num for num, j in  enumerate(rows_ids_flat) for k in range(j.shape[-1])]
            columns_flat_pos = [num for num, j in  enumerate(columns_ids_flat) for k in range(j.shape[-1])]

            #记录孩子节点编号
            id_node_row = {}
            count_num = 0
            for i, level in enumerate(self.tables[idx]['row_subtree']):
                for j, node in enumerate(level):
                    id_node_row[node + "_" + str(i)] = count_num
                    count_num += 1
            parents_row = {i:[i] for i in range(len(id_node_row))} 
            for i, level in enumerate(self.tables[idx]['row_subtree']):
                for j, node in enumerate(level):
                    for k, child in enumerate(level[node]):
                        parents_row[id_node_row[[k for k in child][0]+ "_" + str(i + 1)]].append(id_node_row[node + "_" + str(i)])
            
            #记录孩子节点编号
            id_node_column = {}
            count_num = 0
            for i, level in enumerate(self.tables[idx]['column_subtree']):
                for j, node in enumerate(level):
                    id_node_column[node+ "_" + str(i)] = count_num
                    count_num += 1
            parents_column = {i:[i] for i in range(len(id_node_column))} 
            for i, level in enumerate(self.tables[idx]['column_subtree']):
                for j, node in enumerate(level):
                    for k, child in enumerate(level[node]):
                        parents_column[id_node_column[[k for k in child][0]+ "_" + str(i + 1)]].append(id_node_column[node+ "_" + str(i)])


            if self.mask == "yes":
                # content掩码实现
                # contents_rows_mask = [[torch.where(rows_pos[-1] != i, torch.ones([rows_pos[-1].shape[-1]], dtype=torch.bool), torch.zeros([(rows_pos[-1].shape[-1])], dtype=torch.bool)).unsqueeze(0).repeat(val_j.shape[-1], 1) for j, val_j in enumerate(val_i)] for i, val_i in enumerate(content_ids)]
                contents_rows_mask = [[torch.where(rows_pos[-1] != i, torch.ones([rows_pos[-1].shape[-1]]), torch.zeros([(rows_pos[-1].shape[-1])])).unsqueeze(0).repeat(val_j.shape[-1], 1) == 1 for j, val_j in enumerate(val_i)] for i, val_i in enumerate(content_ids)]
                contents_columns_mask = [[torch.where(columns_pos[-1] != j, torch.ones([columns_pos[-1].shape[-1]]), torch.zeros([columns_pos[-1].shape[-1]])).unsqueeze(0).repeat(val_j.shape[-1], 1)==1 for j, val_j in enumerate(val_i)] for i, val_i in enumerate(content_ids)]
                
                content_ids = torch.cat([torch.cat(i, dim=-1) for i in content_ids], dim=-1).squeeze(0)
                contents_rows_mask = torch.cat([torch.cat(i, dim=0) for i in contents_rows_mask], dim=0)
                contents_columns_mask = torch.cat([torch.cat(i, dim=0) for i in contents_columns_mask], dim=0)

                contents_rows_mask = torch.cat((torch.zeros(contents_rows_mask.shape[0], sum([i.shape[-1] for i in rows_ids[:-1]]), dtype=torch.bool), contents_rows_mask), dim=-1)
                contents_columns_mask = torch.cat((torch.zeros(contents_columns_mask.shape[0], sum([i.shape[-1] for i in columns_ids[:-1]]), dtype=torch.bool), contents_columns_mask), dim=-1)

                # row和column掩码实现
                rows_self_mask = torch.tensor([[0 if j in parents_row[i] else 1 for j in rows_flat_pos] for i in rows_flat_pos], dtype=torch.bool)
                columns_self_mask = torch.tensor([[0 if j in parents_column[i] else 1 for j in columns_flat_pos] for i in columns_flat_pos], dtype=torch.bool)

                # row和column对caption掩码实现
                rows_caption_mask = torch.where(rows_caption_pos.unsqueeze(0).repeat(caption_ids.shape[-1], 1).T==0, torch.zeros(rows_caption_pos.shape[-1], caption_ids.shape[-1]), torch.ones([rows_caption_pos.shape[-1], caption_ids.shape[-1]])) == 1
                columns_caption_mask = torch.where(columns_caption_pos.unsqueeze(0).repeat(caption_ids.shape[-1], 1).T==0, torch.zeros(columns_caption_pos.shape[-1], caption_ids.shape[-1]), torch.ones([columns_caption_pos.shape[-1], caption_ids.shape[-1]])) == 1

                results.update({
                    "caption_ids": caption_ids,
                    "content_ids": content_ids,
                    "contents_rows_mask": contents_rows_mask,
                    "contents_columns_mask": contents_columns_mask,
                    "rows_self_mask": rows_self_mask,
                    "columns_self_mask": columns_self_mask,
                    "rows_caption_mask": rows_caption_mask,
                    "columns_caption_mask": columns_caption_mask
                })

            else:
                content_ids = torch.cat([torch.cat(i, dim=-1) for i in content_ids], dim=-1).squeeze(0)
                results.update({
                    "content_ids": content_ids,
                })
            
            rows_ids = torch.cat(rows_ids, dim=-1).squeeze(0)
            columns_ids = torch.cat(columns_ids, dim=-1).squeeze(0)
            results.update({
                    "rows_ids": rows_ids,
                    "columns_ids": columns_ids
            })

        return results

@dataclass
class DataCollatorForData2TextGeneration:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    model_type:str

    def __call__(
        self, examples) -> Dict[str, torch.Tensor]:
        padding_values = {
            "label": -100,
            "id": self.tokenizer.pad_token_id,
            "mask": False
        }
        batch = {key:[e[key] for e in examples] for key in examples[0].keys()}
        match = {key: [pad for pad in padding_values if pad in key][0] for key in batch}
        batch = {key: self._tensorize_batch(batch[key], padding_values[match[key]]) for key in batch}
        return batch

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]], padding_value
    ) -> torch.Tensor:
        if len(examples[0].shape) == 1:
            max_len = max([e.shape[0] for e in examples])
            if self.model_type == "gpt2":
                examples = [torch.cat((torch.full((max_len - e.shape[0], 1), padding_value).to(e.device).squeeze(-1), e)) for e in examples]
            else:
                examples = [torch.cat((e, torch.full((max_len - e.shape[0], 1), padding_value).to(e.device).squeeze(-1))) for e in examples]
            return pad_sequence(examples, batch_first=True, padding_value=padding_value)
        elif len(examples[0].shape) == 2:
            if examples[0].type() == "torch.BoolTensor":
                max_len = max([e.shape[0] for e in examples])
                examples = [torch.cat((e, torch.full((max_len - e.shape[0], e.shape[1]), padding_value).to(e.device)), dim=0) for e in examples]
            max_len = max([e.shape[1] for e in examples])
            examples = torch.cat([torch.cat((e, torch.full((e.shape[0] ,max_len - e.shape[1]), padding_value).to(e.device)), dim=-1).unsqueeze(0) for e in examples], dim=0)
            return examples
        assert False


class Data2TextEvaluationDataset(Dataset):

    def __init__(self, 
        tokenizer: PreTrainedTokenizer,
        src_path: str, 
        block_size: int, 
        bos_tok:str, 
        eos_tok:str, 
        rewrite="no",
        table="no",
        mask="no",
        metric="BLEU",
        **kwargs):

        assert os.path.isfile(src_path), f"Input file path {src_path} not found"
        print(f"Creating features from dataset file at {src_path}")

        self.tok = tokenizer
        self.block_size =  block_size
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.table = table
        self.rewrite = rewrite
        self.mask = mask

        # 获取文本数据
        src_lines = [i.replace("\n", " ") + bos_tok for i in open(src_path, "r")]
        self.tables = json.load(open(kwargs["table_path"], "r"))

        # 处理负样本

        assert "cand_path" in kwargs and os.path.exists(kwargs["cand_path"])
        src_cand_lines = [[]]
        for cand in open(kwargs["cand_path"], "r"):
            if cand == "\n":
                src_cand_lines.append([])
            else:
                src_cand_lines[-1].append(" {} {}".format(cand.replace("\n", ""), eos_tok))
        src_cand_lines.pop()           

        if "cand_metrics_path" in kwargs:
            src_cand_metrics = json.load(open(kwargs["cand_metrics_path"], "r"))[metric]
            src_cand_lines = [[j for _, j in sorted(zip(src_cand_metrics[i], src_cand_lines[i]), reverse=True)] for i, _ in enumerate(src_cand_lines)]

        if "tgt_path" in kwargs:
            tgt_lines = [[]]
            for tgt in open(kwargs["tgt_path"], "r"):
                if tgt == "\n":
                    tgt_lines.append([])
                else:
                    tgt_lines[-1].append(tgt.replace("\n", ""))
            tgt_lines.pop()
            self.cand_input_lines = [[src_lines[i] + " " + tgt_lines[i][0][:-2]+". " + bos_tok] + [src_lines[i] + cand for cand in cands] for i, cands in enumerate(src_cand_lines)]
        else:
            self.cand_input_lines = [[src_lines[i] + cand for cand in cands] for i, cands in enumerate(src_cand_lines)]
        return


    def __len__(self):
        return len(self.cand_input_lines)

    def __getitem__(self, idx):
        separator = self.tok(self.bos_tok, add_special_tokens=False)['input_ids'][0]
        results = {}
        cand_input_ids = self.tok(self.cand_input_lines[idx], add_special_tokens=True, truncation=True, max_length=self.block_size, is_split_into_words=False, padding=True)["input_ids"]


        cand_masks = copy.deepcopy(cand_input_ids)
        for i, elem in enumerate(cand_input_ids):
            sep_idx = elem.index(separator)
            cand_masks[i][:sep_idx] = [self.tok.pad_token_id] * sep_idx
        results.update({"cand_input_ids" : torch.tensor(cand_input_ids, dtype=torch.int64), "cand_masks" : torch.tensor(cand_masks, dtype=torch.int64)})

        
        if self.table == "yes":            
            caption_ids = self.tok.batch_encode_plus([self.tables[idx]['caption']], max_length=self.block_size, return_tensors="pt", pad_to_max_length=False, truncation=True)
            caption_ids = caption_ids["input_ids"].squeeze(0)

            rows_ids = [[self.tok(j, return_tensors="pt")['input_ids'] for j in  i] for i in self.tables[idx]['row_subtree']]
            columns_ids = [[self.tok(j, return_tensors="pt")['input_ids'] for j in  i] for i in self.tables[idx]['column_subtree']]
            content_ids = [[torch.cat([self.tok(j[v], return_tensors="pt")['input_ids'] for v in j], dim=-1) for j in self.tables[idx]['mp'][i]] for i in self.tables[idx]['mp']]

            rows_ids_flat = [j for i in rows_ids for j in i]
            columns_ids_flat = [j for i in columns_ids for j in i]

            ## row与column对caption准备位置掩码
            rows_caption_pos = torch.cat([torch.cat([torch.tensor([h] * j.shape[-1]) for num, j in  enumerate(i)]) for h, i in enumerate(rows_ids)])
            columns_caption_pos = torch.cat([torch.cat([torch.tensor([h] * j.shape[-1]) for num, j in  enumerate(i)]) for h, i in enumerate(columns_ids)])
            
            ## content掩码准备位置掩码            
            rows_pos = [torch.cat([torch.tensor([num] * j.shape[-1]) for num, j in  enumerate(i)]) for i in rows_ids]
            columns_pos = [torch.cat([torch.tensor([num] * j.shape[-1]) for num, j in  enumerate(i)]) for i in columns_ids]

            rows_ids = [torch.cat(i, dim=-1) for i in rows_ids]
            columns_ids = [torch.cat(i, dim=-1) for i in columns_ids]

            ## row和column掩码准备位置掩码
            rows_flat_pos = [num for num, j in  enumerate(rows_ids_flat) for k in range(j.shape[-1])]
            columns_flat_pos = [num for num, j in  enumerate(columns_ids_flat) for k in range(j.shape[-1])]

            #记录孩子节点编号
            id_node_row = {}
            count_num = 0
            for i, level in enumerate(self.tables[idx]['row_subtree']):
                for j, node in enumerate(level):
                    id_node_row[node + "_" + str(i)] = count_num
                    count_num += 1
            parents_row = {i:[i] for i in range(len(id_node_row))} 
            for i, level in enumerate(self.tables[idx]['row_subtree']):
                for j, node in enumerate(level):
                    for k, child in enumerate(level[node]):
                        parents_row[id_node_row[[k for k in child][0]+ "_" + str(i + 1)]].append(id_node_row[node + "_" + str(i)])
            
            #记录孩子节点编号
            id_node_column = {}
            count_num = 0
            for i, level in enumerate(self.tables[idx]['column_subtree']):
                for j, node in enumerate(level):
                    id_node_column[node+ "_" + str(i)] = count_num
                    count_num += 1
            parents_column = {i:[i] for i in range(len(id_node_column))} 
            for i, level in enumerate(self.tables[idx]['column_subtree']):
                for j, node in enumerate(level):
                    for k, child in enumerate(level[node]):
                        parents_column[id_node_column[[k for k in child][0]+ "_" + str(i + 1)]].append(id_node_column[node+ "_" + str(i)])


            if self.mask == "yes":
                # content掩码实现
                # contents_rows_mask = [[torch.where(rows_pos[-1] != i, torch.ones([rows_pos[-1].shape[-1]], dtype=torch.bool), torch.zeros([(rows_pos[-1].shape[-1])], dtype=torch.bool)).unsqueeze(0).repeat(val_j.shape[-1], 1) for j, val_j in enumerate(val_i)] for i, val_i in enumerate(content_ids)]
                contents_rows_mask = [[torch.where(rows_pos[-1] != i, torch.ones([rows_pos[-1].shape[-1]]), torch.zeros([(rows_pos[-1].shape[-1])])).unsqueeze(0).repeat(val_j.shape[-1], 1) == 1 for j, val_j in enumerate(val_i)] for i, val_i in enumerate(content_ids)]
                contents_columns_mask = [[torch.where(columns_pos[-1] != j, torch.ones([columns_pos[-1].shape[-1]]), torch.zeros([columns_pos[-1].shape[-1]])).unsqueeze(0).repeat(val_j.shape[-1], 1)==1 for j, val_j in enumerate(val_i)] for i, val_i in enumerate(content_ids)]
                
                content_ids = torch.cat([torch.cat(i, dim=-1) for i in content_ids], dim=-1).squeeze(0)
                contents_rows_mask = torch.cat([torch.cat(i, dim=0) for i in contents_rows_mask], dim=0)
                contents_columns_mask = torch.cat([torch.cat(i, dim=0) for i in contents_columns_mask], dim=0)

                contents_rows_mask = torch.cat((torch.zeros(contents_rows_mask.shape[0], sum([i.shape[-1] for i in rows_ids[:-1]]), dtype=torch.bool), contents_rows_mask), dim=-1)
                contents_columns_mask = torch.cat((torch.zeros(contents_columns_mask.shape[0], sum([i.shape[-1] for i in columns_ids[:-1]]), dtype=torch.bool), contents_columns_mask), dim=-1)

                # row和column掩码实现
                rows_self_mask = torch.tensor([[0 if j in parents_row[i] else 1 for j in rows_flat_pos] for i in rows_flat_pos], dtype=torch.bool)
                columns_self_mask = torch.tensor([[0 if j in parents_column[i] else 1 for j in columns_flat_pos] for i in columns_flat_pos], dtype=torch.bool)

                # row和column对caption掩码实现
                rows_caption_mask = torch.where(rows_caption_pos.unsqueeze(0).repeat(caption_ids.shape[-1], 1).T==0, torch.zeros(rows_caption_pos.shape[-1], caption_ids.shape[-1]), torch.ones([rows_caption_pos.shape[-1], caption_ids.shape[-1]])) == 1
                columns_caption_mask = torch.where(columns_caption_pos.unsqueeze(0).repeat(caption_ids.shape[-1], 1).T==0, torch.zeros(columns_caption_pos.shape[-1], caption_ids.shape[-1]), torch.ones([columns_caption_pos.shape[-1], caption_ids.shape[-1]])) == 1

                results.update({
                    "caption_ids": caption_ids,
                    "content_ids": content_ids,
                    "contents_rows_mask": contents_rows_mask,
                    "contents_columns_mask": contents_columns_mask,
                    "rows_self_mask": rows_self_mask,
                    "columns_self_mask": columns_self_mask,
                    "rows_caption_mask": rows_caption_mask,
                    "columns_caption_mask": columns_caption_mask
                })

            else:
                content_ids = torch.cat([torch.cat(i, dim=-1) for i in content_ids], dim=-1).squeeze(0)
                results.update({
                    "content_ids": content_ids,
                })
            
            rows_ids = torch.cat(rows_ids, dim=-1).squeeze(0)
            columns_ids = torch.cat(columns_ids, dim=-1).squeeze(0)
            results.update({
                    "rows_ids": rows_ids,
                    "columns_ids": columns_ids
            })

        return results


@dataclass
class DataCollatorForData2TextEvaluation:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    format_mode: str = 'cat'
    mlm_probability: float = 0.15

    def __call__(
        self, examples) -> Dict[str, torch.Tensor]:
        padding_values = {
            "label": -100,
            "id": self.tokenizer.pad_token_id,
            "mask": False
        }
        batch = {key:[e[key] for e in examples] for key in examples[0].keys()}
        match = {key: [pad for pad in padding_values if pad in key][0] for key in batch}
        batch = {key: self._tensorize_batch(batch[key], padding_values[match[key]]) for key in batch}
        return batch

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]], padding_value
    ) -> torch.Tensor:
        
        if len(examples[0].shape) == 1:
            return pad_sequence(examples, batch_first=True, padding_value=padding_value)
        elif len(examples[0].shape) == 2:
            if examples[0].type() == "torch.BoolTensor":
                max_len = max([e.shape[0] for e in examples])
                examples = [torch.cat((e, torch.full((max_len - e.shape[0], e.shape[1]), padding_value).to(e.device)), dim=0) for e in examples]
            max_len = max([e.shape[1] for e in examples])
            examples = torch.cat([torch.cat((e, torch.full((e.shape[0] ,max_len - e.shape[1]), padding_value).to(e.device)), dim=-1).unsqueeze(0) for e in examples], dim=0)
            return examples
        assert False
