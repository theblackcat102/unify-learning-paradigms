import random
from collections.abc import Mapping
import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin, 
    _torch_collate_batch,
)
from utils import random_spans_noise_mask


@dataclass
class DataCollatorForUL2(DataCollatorMixin):
    """

    Data collator used for UL2

    """
    tokenizer: PreTrainedTokenizerBase
    r_denoising: bool = True
    r_probability: float = 0.25
    r_denoising_config: Tuple[Tuple] = ((3, 0.15),)
    s_denoising: bool = True
    s_probability: float = 0.5
    x_denoising: bool = True
    x_probability: float = 0.25
    x_denoising_config: Tuple[Tuple] = ((32, 0.5), (64, 0.5))
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __post_init__(self):
        self.total_task = [0, 1, 2]
        task_prob = []
        task_prob.append(self.r_probability if self.r_denoising else 0.0)
        task_prob.append(self.s_probability if self.s_denoising else 0.0)
        task_prob.append(self.x_probability if self.x_denoising else 0.0)
        self.task_prob = task_prob
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.tokenizer.bos_token_id

    def assign_task_type(self, batch_size: int):
        '''
            Randomly assign S,R,X to each sentence based on weighted prob
        '''
        return random.choices(self.total_task,weights=self.task_prob, k=batch_size)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # print(examples)
        task_ids = self.assign_task_type(len(examples))
        task_type = torch.tensor(task_ids)
        lengths = torch.tensor([ len(e['input_ids']) for e in examples ], dtype=torch.long)
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, 
                    pad_to_multiple_of=self.pad_to_multiple_of)
            }
        new_batch = {
            "input_ids": torch.zeros(batch['input_ids'].shape, dtype=torch.long), 
            "labels": torch.zeros(batch['input_ids'].shape, dtype=torch.long)
        }

        _, expanded_length = batch['input_ids'].shape
        input_ids = batch["input_ids"]
        r_denoising_idx = task_type == 0
        if r_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[r_denoising_idx]
            # union of different denoising settings
            for (mean_span, noise) in self.r_denoising_config:
                _mask_indices = torch.tensor([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices
            mask_indices = mask_indices & (sub_input_ids != self.pad_token_id)
            labels_mask = ~mask_indices & (sub_input_ids != self.pad_token_id)

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.to(torch.int16), 'pt')
            labels_sentinel = self.create_sentinel_ids(labels_mask.to(torch.int16), 'pt')

            sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel, 'pt')
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel, 'pt', insert_eos=True)
            new_batch['input_ids'][r_denoising_idx] = sub_input_ids
            new_batch['labels'][r_denoising_idx] = _labels

        s_denoising_idx = task_type == 1
        if s_denoising_idx.any():
            sub_input_ids = input_ids[s_denoising_idx]
            _labels = []
            _input_ids = []
            for input_id, len_ in zip(sub_input_ids, lengths[s_denoising_idx]):
                split = max(len_//2, 2)
                diff = expanded_length - split
                _input_ids.append(F.pad(input_id[:split], (0, diff), 'constant', self.pad_token_id))
                past_seq = input_id[split:]
                if past_seq[-1] != self.tokenizer.eos_token_id:
                    past_seq[-1] = self.tokenizer.eos_token_id
                _labels.append(F.pad(past_seq, (0, split), 'constant', self.pad_token_id))

            new_batch['input_ids'][s_denoising_idx] = torch.stack(_input_ids)
            new_batch['labels'][s_denoising_idx] = torch.stack(_labels)


        x_denoising_idx = task_type == 2
        if x_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[x_denoising_idx]
            for (mean_span, noise) in self.x_denoising_config:
                _mask_indices = torch.tensor([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices
            mask_indices = mask_indices & (sub_input_ids != self.pad_token_id)
            labels_mask = ~mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.to(torch.int16), 'pt')
            labels_sentinel = self.create_sentinel_ids(labels_mask.to(torch.int16), 'pt')

            sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel, 'pt')
            labels = self.filter_input_ids(sub_input_ids, labels_sentinel, 'pt')
            new_batch['input_ids'][x_denoising_idx] = sub_input_ids
            new_batch['labels'][x_denoising_idx] = labels

        return self.prepare_decoder_inputs_from_labels(new_batch)


    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        task_ids = self.assign_task_type(len(examples))
        task_type = np.array(task_ids)
        lengths = np.array([ len(e['input_ids']) for e in examples ], dtype=np.int32)
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="np", 
                pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, 
                    pad_to_multiple_of=self.pad_to_multiple_of)
            }
        new_batch = {
            "input_ids": np.zeros(batch['input_ids'].shape, dtype=np.int32), 
            "labels": np.zeros(batch['input_ids'].shape, dtype=np.int32)
        }

        _, expanded_length = batch['input_ids'].shape
        input_ids = batch["input_ids"]
        r_denoising_idx = task_type == 0
        if r_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[r_denoising_idx]
            for (mean_span, noise) in self.r_denoising_config:
                _mask_indices = np.array([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices
            mask_indices = mask_indices & (sub_input_ids != self.pad_token_id)
            labels_mask = ~mask_indices & (sub_input_ids != self.pad_token_id)

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int16))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int16))

            sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel)
            new_batch['input_ids'][r_denoising_idx] = sub_input_ids
            new_batch['labels'][r_denoising_idx] = _labels

        s_denoising_idx = task_type == 1        
        if s_denoising_idx.any():
            sub_input_ids = input_ids[s_denoising_idx]
            _labels = []
            _input_ids = []
            for input_id, len_ in zip(sub_input_ids, lengths[s_denoising_idx]):
                split = max(len_//2, 2)
                diff = expanded_length - split
                _input_ids.append(np.pad(input_id[:split], (0, diff), 'constant'))
                past_seq = input_id[split:]
                if past_seq[-1] != self.tokenizer.eos_token_id:
                    past_seq[-1] = self.tokenizer.eos_token_id
                _labels.append(np.pad(past_seq, (0, split), 'constant'))

            new_batch['input_ids'][s_denoising_idx] = np.array(_input_ids)
            new_batch['labels'][s_denoising_idx] = np.array(_labels)


        x_denoising_idx = task_type == 2
        if x_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[x_denoising_idx]
            for (mean_span, noise) in self.x_denoising_config:
                _mask_indices = np.array([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices
            mask_indices = mask_indices & (sub_input_ids != self.pad_token_id)
            labels_mask = ~mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int16))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int16))

            sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            labels = self.filter_input_ids(sub_input_ids, labels_sentinel)
            new_batch['input_ids'][x_denoising_idx] = sub_input_ids
            new_batch['labels'][x_denoising_idx] = labels

        return self.np_prepare_decoder_inputs_from_labels(new_batch)

    def filter_input_ids(self, input_ids, sentinel_ids, type='np', insert_eos=False):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = []
        for row in input_ids_full:
            collapsed_id = row[row >= 0]
            diff = len(row) - len(collapsed_id)
            collapsed_id = np.pad(collapsed_id, (0, diff), 'constant')
            if insert_eos:
                collapsed_id[diff] = self.tokenizer.eos_token_id
            input_ids.append(collapsed_id)
        if type == 'pt':
            return torch.from_numpy(np.array(input_ids))
        return np.array(input_ids)

    def create_sentinel_ids(self, mask_indices, type='np'):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        if type == 'pt':
            mask_indices = mask_indices.numpy()

        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices
        if type == 'pt':
            return torch.from_numpy(sentinel_ids)
        return sentinel_ids

    def prepare_decoder_inputs_from_labels(self, batch):
        # decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id.
        # See T5 docs for more information
        batch["labels"][ batch["labels"] == self.pad_token_id ] = self.label_pad_token_id
        shifted_labels = batch["labels"].new_zeros(batch["labels"].shape)
        shifted_labels[..., 1:] = batch["labels"][..., :-1].clone()
        shifted_labels[..., 0] = self.decoder_start_token_id  # decoder_start_token_id

        batch["decoder_input_ids"] = torch.masked_fill(
            shifted_labels,
            shifted_labels == self.label_pad_token_id,
            self.pad_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            shifted_labels == self.label_pad_token_id,
            0,
            torch.ones_like(shifted_labels),
        )
        return batch

    def np_prepare_decoder_inputs_from_labels(self, batch):
        batch["labels"][ batch["labels"] == self.pad_token_id ] = self.label_pad_token_id
        shifted_labels = np.zeros(batch["labels"].shape)
        shifted_labels[..., 1:] = batch["labels"][..., :-1].copy()
        shifted_labels[..., 0] = self.decoder_start_token_id

        batch["decoder_input_ids"] = np.where(
            shifted_labels == self.label_pad_token_id,
            self.pad_token_id,
            shifted_labels
        )
        batch["decoder_attention_mask"] = np.where(
            shifted_labels == self.label_pad_token_id,
            0,
            np.ones_like(shifted_labels)
        )
        return batch

