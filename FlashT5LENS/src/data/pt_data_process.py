"""
Input:
    - filtered 'train' json files. 
    mainly for generating POP and NML pre-training tasks.
    ['path', 'label1', 'label2' ,'label3', dataset_type:'train', text:"", network_model_layer:""]
    - The text is the hexdecimal representation of the pcap file. -with desired length truncated by pcap2json.
Output:
    - pt_datasets object and use 'data_collator_pt' to collate with mask setting.
    - Tokenized pre-training data into specific length for training.
    - Generate pre-training task labels.

Can't read each pcap from the disk, that's too slow.
pcaps should have been organized into json files.
"""
import os
import numpy as np
import copy
from itertools import groupby
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import List
import hashlib



def _trunc_sanity_check(tokenizer, truncted_content:List):
    """
    Using bert tokenizer to check whether truncation is valid or not:
    need to dispose non-complete subwords casued by truncation
    tokenizer decode back to see original input:
    background: 
    e.g. "unbelievable"->['un', '##believable'] 2 subwords
    e.g. "unaffable"->["un", "#aff", '##able'] 3 subwords
    """
    if len(truncted_content) > 0:
        str = tokenizer.decode(truncted_content[-1])
        if '#' not in str and len(str) < 4: # [dispose next subword] => The input is 4 digit hexadecimal number
            truncted_content = truncted_content[:-1]
        elif '#' in str:
            str2 = tokenizer.decode(truncted_content[-2])
            str2 = str2 + str.replace("#", "") # [combine two subwords] ->can be [unbelievable or #affable] both are valid
            if '#' not in str2 and len(str2) < 4:
                truncted_content = truncted_content[:-2]
    return truncted_content
        


class pt_dataset:
    # pt_data_names = ['DoHBrw_train_flow', 'VPN_train_flow', 'USTC-TFC2016_train_flow', 'IoT-Lite_train_flow', 'Tor_train_flow']
    pt_data_names = ['USTC-TFC2016_val_flow']
    
    def __init__(self, 
        tokenizer:AutoTokenizer,
        pt_data_path:str,
        pkt_per_flow:int,
        optim_len:int,
        pop_percent:float,
        pop_switch_gap:int,
        nml_label_gap:int,
    )-> Dataset:
        """
        Args:
            tokenizer: AutoTokenizer object.
            pt_data_path: Path to the pre-training data.
            pkt_per_flow: Number of packets per flow.
            optim_len: The inflated optimal input length, considering the MSP tasks will delete some tokens. The final input length will like the setted length.
            pop_percent: Percent of data samples involved in POP pretrain tasks.
            pop_switch_gap: Gap between pop switch operations.
            nml_label_gap: Gap between next nml label prediction.
        """
        super().__init__()
        self.tokenizer = tokenizer
        # Check the special tokens not <unk>(2) in wordpiece tokenizer.
        pt_dataset.EOS_ID = self.tokenizer.eos_token_id             # 1
        pt_dataset.HEAD_ID = self.tokenizer.convert_tokens_to_ids('<head>') # 3
        pt_dataset.PKT_ID = self.tokenizer.convert_tokens_to_ids('<pkt>') # 4
        self.optim_len = optim_len
        self.pkt_per_flow = pkt_per_flow
        self.pop_percent = pop_percent
        self.pop_switch_gap = pop_switch_gap
        self.nml_label_gap = nml_label_gap
        self.pt_data_path = pt_data_path
        self.cache_dir = '/data/lcharles/FAT5LENS/cache/pt_toy_val'
        ...
        
    def load_single_dataset(self, pt_data_name):
        print(f"Loading {pt_data_name} dataset...")
        pth = os.path.join(self.pt_data_path, pt_data_name.lower() + '.json')
        pt_data = load_dataset('json', data_files=pth)
        return pt_data['train'].select_columns(['text', 'network_model_layer'])

    def generate_cache_file_name(self):
        # Create a unique cache file name based on input parameters
        params = f"{self.pt_data_names}{self.optim_len}{self.pkt_per_flow}{self.pop_percent}{self.pop_switch_gap}{self.nml_label_gap}"
        return os.path.join(self.cache_dir, f"dataset_cache_{hashlib.md5(params.encode()).hexdigest()}.arrow")


    def load_and_process(self)->Dataset:
        '''
            Load pre-training data from json files based on pt_datanames.
            Return combined pre-training dataset.
        '''
        
        # Use ProcessPoolExecutor for parallel dataset loading
        with ProcessPoolExecutor(max_workers=min(len(self.pt_data_names), multiprocessing.cpu_count())) as executor:
            pt_datasets = list(executor.map(self.load_single_dataset, self.pt_data_names))

        pt_dataset = concatenate_datasets(pt_datasets)

        cache_file_name = self.generate_cache_file_name()
        if os.path.exists(cache_file_name):
            print(f"Loading cached dataset from {cache_file_name}")
            return load_dataset('arrow', data_files=[cache_file_name])['train']


        self.pt_dataset = pt_dataset.map(
            self.tokenize_pt_func,
            batched=True, # 1000 samples per batch
            fn_kwargs={
                'tokenizer':self.tokenizer,
                'optim_len':self.optim_len,
                'pkt_per_flow':self.pkt_per_flow,
                'pop_percent':self.pop_percent,
                'pop_switch_gap':self.pop_switch_gap,
                'nml_label_gap':self.nml_label_gap,
            },
            remove_columns=['text', 'network_model_layer'],
            num_proc=8,
            cache_file_name=cache_file_name,
        )
        
        self.pt_dataset = self.pt_dataset.shuffle(seed=42)
        
        return self.pt_dataset
    
    def _save_pt_dataset():
        
        ...
    
    # Set it as static method in case of mixing with other class parameters.
    @staticmethod   
    def tokenize_pt_func(example,
                         tokenizer:AutoTokenizer,
                         optim_len:int,
                         pkt_per_flow:int,
                         pop_percent:float,
                         pop_switch_gap:int,
                         nml_label_gap:int,
                         )->dict:
        """
            The tokenization process for pre-training data.
            The POP and NML tasks are before MSP tasks.
            1. POP -> NML -> MSP 
            [Packet order change leads to network model layer label change]
            2. Then, MSP are applied to the switched packets.
    
            Args:
                example: The example to be tokenized.
                tokenizer: The tokenizer object.
                optim_len: The optimal length of the input. If we input this lenght, after MSP, it will become the setted length.
                pkt_per_flow: The number of tokens for each packet.
                nml_label_size: The size of the network model layer label.
                POP_percent: The percentage of POP pretrain tasks.
                POP_switch_gap: The gap between POP switch operations.
        """
        tknzed_input = tokenizer(text=example['text'], return_attention_mask=False)['input_ids']
        
        input_index = np.arange(len(tknzed_input))
        
        pop_index = np.random.choice(input_index, int(len(input_index)*pop_percent), replace=False)
        pop_order = np.full((len(input_index), pkt_per_flow), -100, dtype=np.int8)
        
        # split nml_labels by ';' and convert to int
        # each packet of a flow has the num of 'nml_label_gap' labels [physical, data, network, transport] layers
        # ["0,2,3,2;0,2,3,2", "0,2,3,2"] -> [[0,2,3,2,0,2,3,2], [0,2,3,2]] -> pad with -100
        nml_labels = np.full((len(input_index), pkt_per_flow * nml_label_gap), -100, dtype=np.int8)
        nml_actual_labels  = [np.concatenate([eval(single_pkt_label) for single_pkt_label in item.split(';')]) for item in example['network_model_layer']]
        for i, actual_label in enumerate(nml_actual_labels):
            nml_labels[i, :len(actual_label)] = actual_label 
        
        for id_input, input_text in enumerate(tknzed_input):
            current_len = len(input_text)
            packets = [list(group) for key, group in groupby(input_text, key=lambda k: k != pt_dataset.PKT_ID) if key]
            if packets[-1][0] == pt_dataset.EOS_ID and len(packets[-1]) == 1:
                packets.pop()
            packet_num = len(packets)
            
            if current_len > optim_len:
                # Need to truncate the input to optim_len - packet_num - eos_num(1)
                # proportionally truncate each packet based on the length of each packet. Trucate the payload first and then header.
                optim_len_wo_special_tokens = optim_len - packet_num - 1
                delete_len = current_len - optim_len_wo_special_tokens
                
                header_payload = [{'header': pkt[:pkt.index(pt_dataset.HEAD_ID)],
                                   'raw': pkt[pkt.index(pt_dataset.HEAD_ID)+1:]} for pkt in packets]
                total_payload_len = sum(len(hd_pd['raw']) for hd_pd in header_payload)
                total_header_len = sum(len(hd_pd['header']) for hd_pd in header_payload)
                
                if total_payload_len >= delete_len:
                    # cal ratio and delete from payload
                    ratio = [len(pkt['raw']) / total_payload_len for pkt in header_payload]
                    for id_pkt, pkt in enumerate(header_payload):
                        pkt['raw'] = pkt['raw'][:-np.ceil(int(delete_len * ratio[id_pkt]))] # delete from the end
                        
                        # Using wordpiece tokenizer needs to check the tokenization problem.
                        pkt['raw'] = _trunc_sanity_check(tokenizer, pkt['raw']) 
                else:
                    # cautious: cal ratio and delete from header
                    delete_len -= total_payload_len
                    ratio = [len(pkt['header']) / total_header_len for pkt in header_payload]
                    for id_pkt, pkt in enumerate(header_payload):
                        pkt['header'] = pkt['header'][:-np.ceil(int(delete_len * ratio[id_pkt]))]
                        print("Warning: It is cutting the header, now.")
                        # Using wordpiece tokenizer needs to check the tokenization problem.
                        pkt['header'] = _trunc_sanity_check(tokenizer, pkt['header']) 
                        # empty payload
                        pkt['raw'] = []
                
                # reconstruct the input and save to the tokenized input               
                packets = [packet['header'] + [pt_dataset.HEAD_ID] + packet['raw'] for packet in header_payload]
            
                        
            # construct the pop order
            if id_input in pop_index:
                packet_order = np.arange(packet_num)
                packet_order_permute = copy.deepcopy(packet_order)
               
                if packet_num <= pop_switch_gap:
                   # keep the order still
                    pass
                else:
                   first_cand = np.random.randint(0, (packet_num - pop_switch_gap))
                   second_cand = first_cand + pop_switch_gap
                   packet_order_permute[first_cand], packet_order_permute[second_cand] = packet_order[second_cand], packet_order[first_cand]
                
                pop_order[id_input, :len(packet_order_permute)] = packet_order_permute
            
                packets = [packets[order] for order in packet_order_permute]
            
            
            packets = [pkt + [pt_dataset.PKT_ID] for pkt in packets]
            tknzed_input[id_input] = sum(packets, [])
            assert len(tknzed_input[id_input]) <= optim_len - 1, "The input length is larger than the optimal length." 

        # Padding the input to the optimal length
        tknzed_input = np.array([np.array(i + [-100] * (optim_len - len(i))) for i in tknzed_input])
        return {'input_ids': tknzed_input, 'pop_order': pop_order, 'nml_labels': nml_labels}
        



if __name__ == "__main__":
    # # Load the pre-trained model and tokenizer.
    def _get_tokenizer():

        # tokenizerName = "/data2/charles/Tokenizer/NetT5WordPiece65536"
        tokenizerName = '/data/lcharles/DATASETS/RAWPACAP/Tokenizer/NetBPE32000'
        print(f"Using tokenizer from {tokenizerName}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizerName,
            use_fast=True
        )
        tokenizer.model_max_length = int(1e9)
        return tokenizer
    tokenizer = _get_tokenizer()
    pt_data_path = "/data/lcharles/DATASETS/NETBENCH/netbench_v12/json/flow"
    
    dataset = pt_dataset(tokenizer, pt_data_path, pkt_per_flow=10, optim_len=1024, pop_percent=0.2, pop_switch_gap=5, nml_label_gap=4)
    print(dataset.EOS_ID)
    print(dataset.HEAD_ID)
    print(dataset.PKT_ID)
    ...