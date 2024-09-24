import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import BatchEncoding
# from data.collator import LongContextDataCollatorForT5MLM
from transformers.data.data_collator import DataCollatorMixin
from typing import List, Dict
import numpy as np
import torch

class DataCollatorForPretrain(DataCollatorMixin):
    def __init__(self, 
                 tokenizer:AutoTokenizer,
                 max_length:int,
                 optimal_len:int,   
                 max_labels_length:int,
                 noise_density:float,
                 mean_noise_span_length:int,
                 min_mask_span_length:int=5,
                 ):
        super().__init__()
        """Args:
            noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        """
        
        self.tokenizer = tokenizer
        self.PAD_ID = self.tokenizer.pad_token_id # 0
        self.EOS_ID = self.tokenizer.eos_token_id # 1
        self.HEAD_ID = self.tokenizer.convert_tokens_to_ids("<head>") # 3
        self.PKT_ID = self.tokenizer.convert_tokens_to_ids("<pkt>") # 4
        
        self.max_length = max_length
        self.optimal_len = optimal_len
        self.max_labels_length = max_labels_length
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.min_mask_span_length = min_mask_span_length
        
        # self.optimal_len, self.target_len = self.compute_input_and_target_lengths(
        #     inputs_length=self.max_length,
        #     noise_density=self.noise_density,
        #     mean_noise_span_length=self.mean_noise_span_length,
        # )
            
        ...
    @staticmethod
    def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

        [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
        modified by charles Li.
        Training parameters to avoid padding with random_spans_noise_mask.
        When training a model with random_spans_noise_mask, we would like to set the other
        training hyperparmeters in a way that avoids padding.
        This function helps us compute these hyperparameters.
        We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
        and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
        This function tells us the required number of tokens in the raw example (for split_tokens())
        as well as the length of the encoded targets. Note that this function assumes
        the inputs and targets will have EOS appended and includes that in the reported length.

        Args:
            inputs_length: an integer - desired length of the tokenized inputs sequence
            noise_density: a float
            mean_noise_span_length: a float
        Returns:
            tokens_length: length of original text in tokens
            targets_length: an integer - length in tokens of encoded targets sequence
        """

        def _tokens_length_to_inputs_length_targets_length(tokens_length):
            num_noise_tokens = int(round(tokens_length * noise_density))
            num_nonnoise_tokens = tokens_length - num_noise_tokens
            num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
            # inputs contain all nonnoise tokens, sentinels for all noise spans
            # and one EOS token.
            _input_length = num_nonnoise_tokens + num_noise_spans + 1 # real input length
            #XXX Modified by Charles
            _output_length = num_noise_tokens + num_noise_spans + num_noise_spans-1 + 1 # real output length
            # It is possible that the truncation is sparse (short span) and needs more special token to mask, which will increase the span. 
            # It probably becomes more obvious when only truncating over length > 5.
            return _input_length, _output_length

        tokens_length = inputs_length 

        # if this is smaller than the max input length, we may continue to maximize the real input maximun size
        # until the processed input size reach to the max input length
        while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
            tokens_length += 1

        inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

        # minor hack to get the targets length to be equal to inputs length
        # which is more likely to have been set to a nice round number.
        if noise_density == 0.5 and targets_length > inputs_length:
            tokens_length -= 1
            targets_length -= 1
        return tokens_length, targets_length

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        
        batch = BatchEncoding({
                key: np.array([examples[i][key] for i in range(len(examples))])
                for key, _ in examples[0].items()
            })
        
        input_ids = batch['input_ids']
        real_len = np.sum(input_ids != -100, axis=1)
        
        # MSP task organization
        attn_mask = np.ones_like(input_ids)
        # skip special tokens to mask: make <pkt> and <head> unavailable to mask
        attn_mask[(input_ids ==self.PKT_ID) | (input_ids == self.HEAD_ID)] = 0
        
        spans_noise_masks = np.asarray(
            [
                np.pad(
                    self.segment_spans_random_mask(attn_mask[i][:real_len[i]]),
                    (0, self.optimal_len-real_len[i]),
                    'constant',
                    constant_values=False
                )
                for i in range(len(input_ids))
            ]
        )
        
        # spans_noise_mask [False True False False False]
        input_sentinel = self.create_sentinel_ids(spans_noise_masks.astype(int))
        label_spans_noise_masks = ~spans_noise_masks
        label_sentinel = self.create_sentinel_ids(label_spans_noise_masks.astype(int))
        
        batch['input_ids'] = self.filter_input_ids(input_ids, input_sentinel, self.max_length)
        batch['labels'] = self.filter_input_ids(input_ids, label_sentinel, self.max_labels_length, is_label=True)
        
        # padding mask can be initialized in models.
        
        # Packet order prediction task
        pop_mask = np.zeros_like(batch['input_ids'])
        condition = (batch["input_ids"] == self.PKT_ID) & ((batch["pop_order"][:] != -100).sum(axis=1)!=0)[:, np.newaxis]
        pop_mask[condition] = True
        batch['pop_mask'] = pop_mask.astype(bool)
        
        # Network model layer prediction
        nml_mask = np.zeros_like(batch['input_ids'])
        condition = (batch['input_ids']==self.HEAD_ID) & ((batch["nml_labels"] != -100).sum(axis=-1)!=0)[:, np.newaxis]
        nml_mask[condition] = True
        batch['nml_mask'] = nml_mask.astype(bool)
        
        # TODO: The truncation of input_ids truncted the last <head>, need to put it back.
        assert sum(sum(nml_mask)) == int(sum(batch['nml_labels'][batch['nml_labels']!=-100]>=0)/4), "problem is here, the size of label and mask are not equal"
        
        # process for the header and payload segment embedding
        # Redundency in packet_embed_ids is okay, as these -100 will be masked.
        pkt_token_mask = np.zeros_like(batch['input_ids'])
        pkt_token_mask[batch['input_ids'] == self.PKT_ID] = 1
        pkt_seg_index = np.cumsum(pkt_token_mask, axis=-1) - pkt_token_mask
        batch['pkt_seg_ind'] = pkt_seg_index.astype(np.int64) # [0000,1111,22..,3333,...]
        
        # [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1] ^ [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
        # ->[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
        seg_token_mask = np.zeros_like(batch['input_ids'])
        seg_token_mask[(batch["input_ids"]==self.HEAD_ID)|(batch["input_ids"]==self.PKT_ID)] = 1
        flipped_seg_mask = self.flip_segment_id(seg_token_mask)
        head_payload_seg_ind = flipped_seg_mask ^ seg_token_mask
        batch['head_payload_seg_ind'] = head_payload_seg_ind.astype(np.int64)
        
        # check the input_ids length
        if batch["input_ids"].shape[-1] != self.max_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )
        
        if batch["labels"].shape[-1] != self.max_labels_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.max_labels_length}."
            )

        # Transform from numpy to torch tensors
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}

        return batch
        ...

    def segment_spans_random_mask(self, attn_mask):
        """
        skip special tokens to randomly mask the input
        Input:
                attention mask is [1,1,1,1,0,1,1,1,0,1] # 0 is special token
        Output:
                final mask based on the input mask
        """
        # 1. pad leftmost with 0 and compare with 0 to cumsum
        segment_id = np.cumsum(
            (np.pad(attn_mask, (1, 0))==0).astype(int)
        )
        
        # 2. get the unique segment id
        _, segment_len = np.unique(segment_id, return_counts=True)
        segment_len -= 1
        
        # 3. start to random mask
        final_mask = np.array([], dtype=bool)

        for _, seg_len in enumerate(segment_len):
            final_mask = np.concatenate([final_mask, np.array([False])])
            
            # if the segment length is larger than 5, we will mask it
            if seg_len >= self.min_mask_span_length:
                mask = self.random_spans_noise_mask(seg_len)   
                final_mask = np.concatenate([final_mask, mask])
            else:
                final_mask = np.concatenate([final_mask, np.array([False]*seg_len)])     
        
        ...
        assert final_mask.shape[0] == attn_mask.shape[0] + 1, f"The final mask length {final_mask.shape[0]} is not equal to the input mask length {attn_mask.shape[0]}"

        return final_mask[1:].astype(bool)
    
    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0  [num_noise_token]
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )
        # [clean_length_1, noisy_length_1, clean_length_2, noisy_length_2, ...]
        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]   

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        extra_start_token_id = 107
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (extra_start_token_id - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids, set_length, is_label=False):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        if is_label == False: 
            input_ids_shrink = [item[item>=0] for item in input_ids_full]
            padded_arrays = [np.pad(
                            np.pad(arr,(0, 1), constant_values=self.EOS_ID), # add EOS token
                            (0, set_length - len(arr) - 1), constant_values=self.PAD_ID) for arr in input_ids_shrink] # pad the rest of the sequence
        else:
            # remove the last sentinetal token for labels
            input_ids_shrink = [item[item>=0][:-1] for item in input_ids_full]  
            padded_arrays = [np.pad(
                            np.pad(arr,(0, 1), constant_values=self.EOS_ID),
                            (0, set_length - len(arr) - 1), constant_values=-100) for arr in input_ids_shrink]
                
        input_ids = np.array(padded_arrays)
        return input_ids

    @staticmethod
    def flip_segment_id(input_matrix):
        # input is usually a matrix where <head> and <pkt> are set to 1, others remain 0
        # say [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]
        # say [., ., h, ., ., ., p, ., ., h, ., p]
        # for convenience, we only care about one row value.
        output_matrix = np.ones_like(input_matrix) # now, output is initialized as ones with the shape equal to input_matrix, say [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        for row in range(input_matrix.shape[0]):
            for col in range(input_matrix.shape[1]):
                if input_matrix[row, col] == 1: # if current token is <head> or <pkt>
                    output_matrix[row, col:] = 1 - output_matrix[row, col:] # flip over the rest of `row` in output matrix, for example, suppose row == 1 and col == 2, and current output_maxtrix[1] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], input[1][2] == 1, now output[1][2:] will flip over and turn into output[1][2:] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. So on so forth.
        # after that, output will be [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
        return output_matrix



#TODO: Need to move clsft collator and generate collator here.




if __name__ == '__main__':
    def _get_tokenizer():
        tokenizerName = "/data2/charles/Tokenizer/NetT5WordPiece65536"
        # tokenizerName = '/data/lcharles/DATASETS/RAWPACAP/Tokenizer/NetBPE32000'
        print(f"Using tokenizer from {tokenizerName}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizerName,
            use_fast=True
        )
        tokenizer.model_max_length = int(1e9)
        return tokenizer
    tokenizer = _get_tokenizer()
    
    examples = [
    {'input_ids': np.array([1, 2, 3,4,5,6,7,8 -100, -100]), 'attention_mask': np.array([1, 1, 1, 1,1,1,1,1,0, 0]),
     'pkt_order': np.array([1, 2, 3, -100, -100]), 'nml_label': np.array([1, 2, 3,4,5,6,7 -100, -100])},
    {'input_ids': np.array([4, 5, 6,7,8,9,10,11, -100, -100]), 'attention_mask': np.array([1, 1, 0, 0, 0]),
     'pkt_order': np.array([4, 5, -100, -100, -100]), 'nml_label': np.array([4, 5, -100, -100, -100])}
    ]

    batch = DataCollatorForPretrain(tokenizer, 10, 10,0.2,3, 2)(examples)
    print(f"Batch: {batch}")
