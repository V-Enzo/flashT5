import os
import datasets
from datasets import load_dataset, IterableDataset
from .collator import DataCollatorForCLSFT, DataCollatorForGENFT
from torch.utils.data import DataLoader
from omegaconf import open_dict
from collections import defaultdict
import numpy as np


class LabelMapper:
    def __init__(self,):
        self.label_s2i = {}
        self.counter = defaultdict(int)


    def map_labels2i(self, example, flag):
        filter_label = [item for item in filter(lambda x: x != 0, example['labels'])]
        str_label = " ".join([str(item) for item in filter_label])
        if str_label not in self.label_s2i:
            if flag=='test':
                assert True, "Test Label not in training set"
            self.label_s2i[str_label] = len(self.label_s2i)
        example['labels'] = [self.label_s2i[str_label]]
        if flag == 'train':
            self.counter[self.label_s2i[str_label]] += 1
        return example


    def map_strlabel_counter(self, example, flag):
        filter_label = [item for item in filter(lambda x: x != 0, example['labels'])]
        str_label = " ".join([str(item) for item in filter_label])
        if str_label not in self.counter and flag=='test':
            assert True, "Test Label not in training set"
        if flag == 'train':
            self.counter[str_label] += 1 
        return example


class load_ft_dataloader:
    def __init__(self, tokenizer, args) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self._init_data_path_task_name()
        self._get_train_test_file_path()
        self._load_train_test_datset()
        
        if self.args.task.type == 'classification':
            self._load_dataloader_for_classification()
        elif self.args.task.type == 'clsbytoken':
            self._get_label_distribution()
            self._get_label_weights_on_counter()
            self._get_label_start_id()
            self._load_dataloader_for_clsbytoken()
        elif self.args.task.type == 'generation':
            self._load_dataloader_for_generation()
        else:
            raise NotImplementedError
        self._set_dataloaders()


    def _init_data_path_task_name(self):
        if self.args.task.type == 'classification' or self.args.task.type == 'clsbytoken':
            self.base_dir = os.path.join(self.args.data.base_dir, 'cls')
            self.task_name = self.args.task.name
            self.task_level = self.args.task.level
        
        if self.args.task.type == 'generation':
            self.base_dir = os.path.join(self.args.data.base_dir, 'gen')
            self.task_name = self.args.task.dataset + ' ' + self.args.task.name
            assert self.args.task.level == 'pkt', "generation is only on packet level."
            self.task_level = 'pkt'
            
        self.task_name_list = self.task_name.split('+')
          
            
    def _get_train_test_file_path(self):
        train_dir = os.path.join(self.base_dir, self.task_name_list[0].lower(), 'train')
        train_file_name = '_'.join(self.task_name_list+ [self.task_level] + ['train']).lower()+'.json'
        self.train_path = os.path.join(train_dir, train_file_name)
        
        self.test_path = self.train_path.replace('train', 'test')
        
        self.val_path = self.train_path.replace('train', 'val')



    def _load_train_test_datset(self):
        print("=>Training_file_path:", self.train_path)
        self.train_dataset = load_dataset('json', data_files = self.train_path, split="train")
        print("=>Evaluation_file_path:", self.test_path)
        self.test_dataset = load_dataset('json', data_files= self.test_path, split="train")
        print("=>Validation_file_path:", self.val_path) 
        if os.path.exists(self.val_path):
            self.val_dataset = load_dataset('json', data_files= self.val_path, split="train")        
        else:
            print("=>Validation file not found")
                
                
    def _get_label_distribution(self,):
        self.mapper = LabelMapper()
        datasets.disable_caching()
        self.train_dataset.map(
            self.mapper.map_strlabel_counter,
            batched=False,
            num_proc=1,  # This ensures sequential processing
            fn_kwargs={'flag':"train"}
        )
        
        
        # Check whether the test label exists in train label
        self.test_dataset.map(
            self.mapper.map_strlabel_counter,
            batched=False,
            num_proc=1,  # This ensures sequential processing
            fn_kwargs={'flag':"test"}
        )
        

    def _get_label_weights_on_counter(self,):
        """
        Compute class weights based on frequency.
        
        Args:
        label_counter (Counter or dict): A counter or dictionary where keys are label strings and values are their occurrences in the dataset
        
        Returns:
        dict: A dictionary where keys are label strings and values are their computed weights
        """
        labels = np.array(list(self.mapper.counter.keys()))
        counts = np.array(list(self.mapper.counter.values()))
        
        # Compute inverse frequency
        total_samples = np.sum(counts)
        class_weights = total_samples / (len(labels) * counts)
        
        # Normalize weights
        class_weights = class_weights / np.sum(class_weights) * len(counts)
        
        # Create a dictionary of label to weight
        label_weight = dict(zip(labels, class_weights))
        
        self.label_weight = label_weight
   
   
    def _get_label_start_id(self,):
        self.label_start_id = tuple(map(int, set([item.split(' ')[0] for item in self.mapper.counter.keys()])))
             
                
    def _load_dataloader_for_classification(self,):
        # To implement add validation dataset
            # Create an instance of LabelMapper
        self.mapper = LabelMapper()

        # Apply the mapping to the dataset sequentially
        datasets.disable_caching()
        self.train_dataset = self.train_dataset.map(
            self.mapper.map_labels2i,
            batched=False,
            num_proc=1,  # This ensures sequential processing
            fn_kwargs={'flag':"train"}
        )

        # Print the label to index mapping
        print("=>Training Label to Index Mapping:", self.mapper.label_s2i)

        # Apply the mapping to the dataset sequentially
        self.test_dataset = self.test_dataset.map(
            self.mapper.map_labels2i,
            batched=False,
            num_proc=1,  # This ensures sequential processing
            fn_kwargs={'flag':"test"}
        )
        print("=> Loading DataCollatorForCLSFT")
        self.data_collator = DataCollatorForCLSFT(
                tokenizer=self.tokenizer,
                label_padding_tokens=-100,
                eos_token_id=self.tokenizer.eos_token_id,
                tsk_type='classification'
            )


    def _load_dataloader_for_clsbytoken(self,):
    
        self.data_collator = DataCollatorForCLSFT(
            tokenizer=self.tokenizer,
            label_padding_tokens=-100,
            eos_token_id=self.tokenizer.eos_token_id,
            tsk_type='clsbytoken'
        )


    def _load_dataloader_for_generation(self,):
        
        print("=> Loading DataCollatorForGENFT")
        self.data_collator = DataCollatorForGENFT(
            tokenizer=self.tokenizer,
            label_padding_tokens=-100,
            eos_token_id=self.tokenizer.eos_token_id,
        )


    def _set_dataloaders(self,):
        is_iterable = isinstance(self.train_dataset, IterableDataset)
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle= not is_iterable,
            collate_fn=self.data_collator,
            batch_size=self.args.optim.batch_size // self.args.optim.grad_acc,
            num_workers=self.args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )    
        self.test_dataloader = DataLoader(
            self.test_dataset,
            shuffle= False,
            collate_fn=self.data_collator,
            batch_size=self.args.optim.batch_size // self.args.optim.grad_acc,
            num_workers=self.args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        if hasattr(self, 'val_dataset'):
            self.val_dataloader = DataLoader(
                self.val_dataset,
                shuffle= False,
                collate_fn=self.data_collator,
                batch_size=self.args.optim.batch_size // self.args.optim.grad_acc,
                num_workers=self.args.data.num_workers,
                pin_memory=True,
                drop_last=False,
            )


if __name__ == '__main__':
    load_ft_dataloader()