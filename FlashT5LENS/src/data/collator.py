from transformers import AutoTokenizer
from transformers import BatchEncoding
from dataclasses import dataclass
import torch
import numpy as np
from typing import List, Dict

STOP_TOKEN_ID = 1
HEADER_TOKEN_ID = 3
PKT_TOKEN_ID = 4
TOKEN_IGNORE_ID = -100

np.random.seed(2137)


@dataclass
class LongContextDataCollatorForT5MLM:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pkt_per_flow: int
    pad_token_id: int
    

    @staticmethod
    def switch_matrix(input_matrix):
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
    
    
    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )
        # print(self.input_length)
        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape
        real_length = self.get_real_length(input_ids, -100)
        input_attention_mask = np.ones_like(input_ids, dtype=np.int8)
        # skip special token to mask, 4 is the special token. <pkt>
        input_attention_mask[input_ids==4] = 0
        # May 15, added mask, 3 is the special token <head> for NML
        input_attention_mask[input_ids==HEADER_TOKEN_ID] = 0 
        mask_indices = np.asarray(
            [
                np.pad(self.modify_skip_special_tokens_to_mask(input_attention_mask[i][:real_length[i]]), (0, expandend_input_length-real_length[i]), 'constant', constant_values=False)
                for i in range(batch_size)
            ]
        )

        labels_mask = ~mask_indices
        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
        
        # truncated the expanded length to desired length
        # batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel, 'input')
        # batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel, 'label')
        
        batch["input_ids"] = self.copy_filter_input_ids(input_ids, input_ids_sentinel, self.input_length)
        batch["labels"] = self.copy_filter_input_ids(input_ids, labels_sentinel, self.target_length, is_label=True)        
        
        # [2024-05-26, attetion mask is the padding mask]
        attention_mask = np.ones_like(batch["input_ids"])
        attention_mask[batch["input_ids"] == self.tokenizer.pad_token_id] = 0
        batch["attention_mask"] = attention_mask

        decoder_attention_mask = np.ones_like(batch["labels"])
        decoder_attention_mask[batch["labels"] == -100] = 0        
        batch["decoder_attention_mask"] = decoder_attention_mask
        
        # Design for POP (packet order prediction)
        pkt_POP_attn_mask = np.zeros_like(batch["input_ids"])
        condition = (batch["input_ids"] == 4) & ((batch["pkt_order"][:] != -100).sum(axis=1)!=0)[:, np.newaxis]
        pkt_POP_attn_mask[condition] = True


        # Design for HTP (Homologous Traffic Prediction)
        # 24-June12, comment out the HTP part
        # pkt_HTP_attn_mask = np.zeros_like(batch["input_ids"])
        # condition = (batch["input_ids"] ==4) & ((batch["same_origin"][:] != -100).sum(axis=1)!=0)[:, np.newaxis]
        # pkt_HTP_attn_mask[condition] = True

        # Design for NML (network model layer prediction)
        pkt_NML_attn_mask = np.zeros_like(batch["input_ids"])
        condition = (batch['input_ids']==HEADER_TOKEN_ID) & ((batch["nml_label"] != -100).sum(axis=-1)!=0)[:, np.newaxis] # each pkt has multiple lists (4 labels for each head)
        pkt_NML_attn_mask[condition] = True
        # TODO: The truncation of input_ids truncted the last <head>, need to put it back.
        assert sum(sum(pkt_NML_attn_mask)) == int(sum(batch['nml_label'][batch['nml_label']!=-100]>=0)/4), "problem is here, the size of label and mask are not equal"


        pkt_embed_mask = np.zeros_like(batch["input_ids"])
        # Redundency in packet_embed_ids is okay, as these -100 will be masked.
        pkt_embed_mask[batch["input_ids"] == 4] = 1
        pkt_embed_ids = np.cumsum(pkt_embed_mask, axis=-1) - pkt_embed_mask
        batch["pkt_embed_ids"] = pkt_embed_ids.astype(np.int64)
        # if  np.max(batch["pkt_embed_ids"]) == 10:
        #         print("it is wrong needs debug.")

        head_embed_mask = np.zeros_like(batch["input_ids"])
        head_condition = (batch["input_ids"] == 3) | (batch["input_ids"] == 4)
        head_embed_mask[head_condition] = 1 # make tokens that are <head> or <pkt> set to 1, say [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]*
        head_embed_ids = self.switch_matrix(head_embed_mask) 
        #   [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]*
        # ^ [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
        # --------------------------------------
        #   [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]

        # *denotes the value means [h, h, <h>, p, p, p, <p>, h, h, <h>, p, <p>]
        # h denotes head token, <h> denotes the special token <head>, p denotes payload token, <p> denote the special token <pkt>
        head_embed_ids ^= head_embed_mask # [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0], now all h and <h> are set to 1
        batch["head_embed_ids"] = head_embed_ids.astype(np.int64)
        batch["pkt_NML_attn_mask"] = pkt_NML_attn_mask.astype(bool)
        batch["pkt_POP_attn_mask"] = pkt_POP_attn_mask.astype(bool)
        # batch["pkt_HTP_attn_mask"] = pkt_HTP_attn_mask.astype(bool)
        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch    
    
    
    def get_real_length(self, input_ids, fill_token_id):
        return np.array([sum(1 for x in my_array if x != fill_token_id) for my_array in input_ids])
    
    
    def skip_special_tokens_to_mask(self, attention_mask):
        """
        attention_mask: [1,1,1,1,1,1]
        """
        # pad return ranks (number of dimensions)
        # pad to only left 
        segment_id = np.cumsum((np.pad(attention_mask, pad_width=((1,0)))==0).astype(int)) # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        # padded to left make it 1 longer and need to subtract
        segment_length -= 1
        final_mask = np.array([], dtype=bool)
        for id, length in enumerate(segment_length): # eg[47, 41, 38, 1]
            # first 4 are not masked, as setting masking span from 5.
            final_mask = np.concatenate((final_mask, np.array([False])))
            if length >= 5:
                mask = self.random_spans_noise_mask(length)
                final_mask = np.concatenate((final_mask, mask))
            else:
                final_mask = np.concatenate((final_mask, np.array([False]*length)))
        assert final_mask.shape[0] == attention_mask.shape[0]+1, f"The length of final_mask is {final_mask.shape[0]}, the length of attention_mask is {attention_mask.shape[0]}"
        return final_mask[1:].astype(bool)
    
    
    def modify_skip_special_tokens_to_mask(self, attention_mask):
        """
        The input attention mask is [1,1,1,1,0,1,1,1,0,1]
        # 0 is special token here that need to skip
        # pad to left and boolify it, then start to mask.
        # for each length, the leftmost one is special token.
        # The padded starting token should be removed before return.
        """
        segment_id = np.cumsum((np.pad(attention_mask, pad_width=((1,0)))==0).astype(int))
        _, segment_length = np.unique(segment_id, return_counts=True)
        segment_length -= 1
        final_mask = np.array([], dtype=bool)
        for _, length in enumerate(segment_length): #   
            # first 4 are not masked, as setting masking span from 5.
            final_mask = np.concatenate((final_mask, np.array([False])))
            # if length == 0:
            #     continue
            # if length == 1:
            #     final_mask = np.concatenate((final_mask, np.array([False]*1)))
            # if length > 1:
            #     mask = self.random_spans_noise_mask(length)
            #     final_mask = np.concatenate((final_mask, mask))
            if length >= 5:
                mask = self.random_spans_noise_mask(length)
                final_mask = np.concatenate((final_mask, mask))
            else:
                final_mask = np.concatenate((final_mask, np.array([False]*length)))
        
        assert final_mask.shape[0] == attention_mask.shape[0]+1, f"The length of final_mask is {final_mask.shape[0]}, the length of attention_mask is {attention_mask.shape[0]}"
        
        return final_mask[1:].astype(bool)
        
    
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

  
    def filter_input_ids(self, input_ids, sentinel_ids, str):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """        
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 should be removed
        # charles: save last token for eos
        if str == 'input':
            trunc_filt_input_list = [line[line != -1][:self.input_length] for line in input_ids_full]
            for id, item in enumerate(trunc_filt_input_list):
                # deal with scenarios that after remove -1, it becomes less than 512
                if item.shape[0] < self.input_length:
                    # print("expanded the trunc_filt_input")
                    remain_len = self.input_length - item.shape[0]
                    trunc_filt_input_list[id] = np.pad(item, (0, remain_len), 'constant', constant_values=-100) 
                elif item.shape[0]>= self.input_length and item[self.input_length-2] > 0 and item[self.input_length-2] != PKT_TOKEN_ID:
                    print("long context MLM In the over long branch")
                    # problem with the pkt token
                    # TODO:
                    # Need to calculate how many pkt tokens in the seq.
                    # To debug whether it is wrong or not.
                    # This is not correct: XXX<pkt>X + <pkt><eos>, need to debug.
                    trunc_filt_input_list[id][self.input_length-2] = PKT_TOKEN_ID
                    trunc_filt_input_list[id][self.input_length-1] = STOP_TOKEN_ID
                
                    # TODO:
                
                    tmp = trunc_filt_input_list[id]
                    if sum(item[item==PKT_TOKEN_ID]==4)!=sum(tmp[tmp==PKT_TOKEN_ID]==4):
                        print("Bug is here need to debug.")
                    
                    assert sum(item[item==PKT_TOKEN_ID]==4)==sum(tmp[tmp==PKT_TOKEN_ID]==4), f"It will generate more pkt token here, which is wrong. original is {item}, modified is {tmp}"
                
                assert trunc_filt_input_list[id].shape[0] == self.input_length, f"The length is not proper. The length is {item.shape[0]}. The item is {item}, The id is {id}\n The input_ids shape is {input_ids_full[id].shape}\n The input ids is {input_ids_full[id]}"
            input_ids = np.array(trunc_filt_input_list)
            input_ids[input_ids == -100] = self.tokenizer.pad_token_id
        elif str == 'label':
            label_list = []
            for id, line in enumerate(input_ids_full):
                if len(line[line>=0]) > self.target_length:
                    # line = line[line>=0][:self.target_length-1]
                    # line[self.target_length-1] = STOP_TOKEN_ID
                    print("The label is not complete")
                    print(len(line[line>=0]))
                    # assert len(line) == self.target_length, f"(id={id}) The length of current flow is larger than the in_length, current length is {len(line[line >= 0])}, current flow is {line[line >= 0]}, current flow is {(self.tokenizer.decode(line[line >= 0]))}"
                    # label_list.append(line)
                    assert True, "The label is not complete and it should not be"
                else: 
                    label_list.append(np.pad(line[line >= 0], (0, self.target_length-len(line[line >= 0])), 'constant', constant_values=-100))
            input_ids = np.array(label_list)
            
        return input_ids


    def copy_filter_input_ids(self, input_ids, sentinel_ids, set_length, is_label=False):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """

        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        if is_label == False: 
            input_ids_shrink = [item[item>=0] for item in input_ids_full]
            padded_arrays = [np.pad(arr, (0, set_length - len(arr)), constant_values=self.tokenizer.pad_token_id) for arr in input_ids_shrink]
        else:
            input_ids_shrink = [item[item>=0][:-1] for item in input_ids_full]  # remove the last sentinetal token for labels
            padded_arrays = [np.pad(arr, (0, set_length - len(arr)), constant_values=TOKEN_IGNORE_ID) for arr in input_ids_shrink]
                
        # padded_arrays = [np.pad(arr, (0, set_length - len(arr)), constant_values=self.tokenizer.pad_token_id) for arr in input_ids_shrink]
        input_ids = np.array(padded_arrays)
        return input_ids



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
    

@dataclass
class DataCollatorForCLSFT:

    tokenizer: AutoTokenizer
    label_padding_tokens: int=-100
    eos_token_id: int=1
    tsk_type: str='clsbytoken'

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )
        # extend batch labels from (2, 113, ) to (2, 114, ), and then replace the 0 with -100
        batch["input_ids"] = batch["input_ids"].astype(np.int64)
        batch["labels"] = batch["labels"].astype(np.int64)
        batch["labels"] = np.pad(batch["labels"], ((0, 0), (0, 1)), 'constant', constant_values=self.label_padding_tokens)  # extend the size for shift label in decoder

        
        
        # [24-Jun-17, attetion mask is the padding mask]
        attention_mask = np.ones_like(batch["input_ids"])
        attention_mask[batch["input_ids"] == self.tokenizer.pad_token_id] = 0
        batch["attention_mask"] = attention_mask

        decoder_attention_mask = np.ones_like(batch["labels"])
        batch["decoder_attention_mask"] = decoder_attention_mask
        
        
        # in generate_way classification, need to mask the 0 to -100
        # If it is the classifier way, need to comment the last line.
        # As label id can be 0, if use 0 to mask, it will cause problem.
        # TODO:Should modify the input table's padding as -100 not 0.
        if self.tsk_type == 'clsbytoken':
            batch["labels"][batch["labels"] == 0] = self.label_padding_tokens
        batch['decoder_attention_mask'][batch['labels'] == self.label_padding_tokens] = 0
        
        
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch
    ...
    
    
    
@dataclass
class DataCollatorForGENFT:

    tokenizer: AutoTokenizer
    label_padding_tokens: int=-100
    eos_token_id: int=1

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )
        # extend batch labels from (2, 113, ) to (2, 114, ), and then replace the 0 with -100
        batch["input_ids"] = batch["input_ids"].astype(np.int64)
        batch["labels"] = batch["labels"].astype(np.int64)
        batch["labels"] = np.pad(batch["labels"], ((0, 0), (0, 1)), 'constant', constant_values=-100)  # extend the size for shift label in decoder
        batch["labels"][batch["labels"] == 0] = -100
        
        # [24-Jun-17, attetion mask is the padding mask]
        attention_mask = np.ones_like(batch["input_ids"])
        attention_mask[batch["input_ids"] == self.tokenizer.pad_token_id] = 0
        batch["attention_mask"] = attention_mask

        decoder_attention_mask = np.ones_like(batch["labels"])
        
        decoder_attention_mask[batch["labels"] == -100] = 0
        batch["decoder_attention_mask"] = decoder_attention_mask
        
        
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch
    ...
    

