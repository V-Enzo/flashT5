import os
import time
import torch
from .utils.logging_utils import Averager, Logger
from tqdm import tqdm
from .model.config_flasht5lens import FlashT5Config
from .model.modeling_flasht5lens import FlashT5ForPretrainTasks
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from accelerate import Accelerator
from torch.optim import AdamW
from .optim.optimizer import AdamWScale, get_lr_scheduler
from accelerate.utils import set_seed

# from .data.pt_dataset import load_pretrain_dataloader
from .data.pt_data_process import pt_dataset
from dataclasses import dataclass, fields
from datasets import load_dataset
from torch.utils.data import DataLoader
from .data.data_collator_pt import DataCollatorForPretrain
from datasets.iterable_dataset import IterableDataset

@dataclass
class PTLossAccuracy:
    msp_loss: float = None
    pop_loss: float = None
    nml_loss: float = None
    msp_acc: float = None
    pop_acc: float = None
    nml_link_acc: float = None
    nml_network_acc: float = None
    nml_transport_acc: float = None
    nml_app_label: float = None
    
    
class pt_trainer:
    def __init__(self, args):
        self.stats = {}
        self.last_log_time = time.time()
        self.train_averager = Averager()
        self.current_train_step = 1
        self._check_args_and_env(args)
        # self._get_config()
        self._set_up_seed_logger_accelerator() #
        self._get_model(load_ckpt=False) # Pretraining from scratch is usually preferred.
        self._get_tokenizer()
        self._load_dataset()
        self._get_optimizer()
        self._get_lr_scheduler()
        self._prepare_compile()
        

    @staticmethod
    @torch.no_grad()
    def calculate_loss_and_accuracy(output):
        results = {}
        if hasattr(output, 'msp_loss'):
            msp_loss = output.msp_loss.item()
            predictions = output.msp_logit.argmax(-1)
            mask = (output.msp_label != -100)
            # For Pretraining, the accuracy is calculated only on the tokens that are not masked
            correct_tokens = (predictions == output.msp_label) & mask
            msp_acc = correct_tokens.sum().item() / mask.sum().item()
            results.update({"msp_loss": msp_loss, "msp_acc": msp_acc})
        
        if hasattr(output, 'pop_loss') and output.pop_loss.item()!=0.0:
            pop_acc = (output.pop_logit.argmax(-1) == output.pop_label).sum().item() / output.pop_label.numel()
            results.update({"pop_loss": output.pop_loss.item(), "pop_acc": pop_acc})
            
        if hasattr(output, 'nml_loss'):
            nml_acc = [] 
            for logit, label in zip(output.nml_logit, output.nml_label):
                nml_acc.append((logit.argmax(-1) == label).sum().item() / label.numel())
            results.update({
                "nml_loss": output.nml_loss.item(),
                "nml_link_acc": nml_acc[0],
                "nml_network_acc": nml_acc[1],
                "nml_transport_acc": nml_acc[2],
                "nml_app_label": nml_acc[3]
            })
            
        return PTLossAccuracy(**results)


    def _check_args_and_env(self, args):
        assert args.optim.batch_size % args.optim.grad_acc == 0
    
        # Train log must happen before eval log
        assert args.eval.every_steps % args.logging.every_steps == 0

        if args.device == 'gpu':
            assert torch.cuda.is_available(), 'We use GPU to train/eval the model'

        assert not (args.eval_only and args.predict_only)

        if args.predict_only:
            assert args.mode == 'ft'
            
        # This lines reduce training step by 2.4x Probably not needed for training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.args = args
        
    
    def _set_up_seed_logger_accelerator(self):
        # set seed and logger, accelerator
        if self.args.seed is not None:
            set_seed(self.args.seed)
            
        self.accelerator = Accelerator(
            cpu=self.args.device == "cpu",
            mixed_precision=self.args.precision,
            log_with="wandb" if self.args.logging.wandb else None,
            gradient_accumulation_steps=self.args.optim.grad_acc,
        )
    
        self.accelerator.init_trackers(
                project_name=self.args.task.name
        )
        
        self.logger = Logger(args= self.args, accelerator=self.accelerator)    
            
            
    # def _get_config(self):
        # config = AutoConfig.from_pretrained(
        #     self.args.model.name,)

        # if hasattr(self.args.model, 'overwrite'):
        #     for k, v in self.args.model.overwrite.items():
        #         assert hasattr(config, k), f'config does not have attribute {k}'
        #         setattr(config, k, v)
        #         print(f'Overwriting config.{k} to {v}')

        # if hasattr(self.args.model, 'add_config'):
        #     for k, v in self.args.model.add_config.items():
        #         assert not hasattr(config, k), f'config already has attribute {k}'
        #         setattr(config, k, v)

        # self.config = config
    
    
    def _load_ckpt(self):
        if self.args.model.restore_path!="":
            path = self.args.model.restore_path
            print("Loading from checkpoint: ", path)
            weights_name = 'pytorch_model.bin'
            input_model_file = os.path.join(path, weights_name)
            print("=>CKPT_FILES", input_model_file)
            ckpt = torch.load(input_model_file, map_location="cuda:0")
            try:
                self.model.load_state_dict(ckpt, strict=True)
                print("=> Successfully loaded checkpoint in strict mode")
            except RuntimeError as e:
                print("=> Error loading checkpoint in strict mode. Error details:")
                print(str(e))
                self.model.load_state_dict(ckpt, strict=False)
                print("=> Successfully loaded checkpoint in non-strict mode")
        else:
            raise ValueError("=>!!!No checkpoint provided, training from scratch")

    
    def _get_optimizer(self):
        no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.optim.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.args.optim.name == 'adamw':
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.optim.base_lr,
            )
        elif self.args.optim.name == 'adamwscale':
            optimizer = AdamWScale(
                optimizer_grouped_parameters,
                lr=self.args.optim.base_lr,
            )
        elif self.args.optim.name == 'adafactor':
            from transformers import Adafactor
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.args.optim.base_lr,
                relative_step=False,
            )
        else:
            raise NotImplementedError

        self.optimizer=optimizer

    
    def _get_model(self, load_ckpt=False):
        self.model = FlashT5ForPretrainTasks(FlashT5Config())
        if load_ckpt:
            self._load_ckpt()
        else:
            print("=>No checkpoint provided, training from scratch")

        # if self.config.is_bf16:
        #     print("=> make model to bfloat16")
        #     self.model = self.model.bfloat16()


    def _get_tokenizer(self):
        if hasattr(self.args.data, 'tokenizer_path'):
            tokenizerName = self.args.data.tokenizer_path
        else:
            tokenizerName = "/data2/charles/Tokenizer/NetT5WordPiece65536"
        print(f"Using tokenizer from {tokenizerName}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizerName,
            use_fast=True
        )
        tokenizer.model_max_length = int(1e9)
        self.tokenizer = tokenizer
        

    def _load_dataset(self):
        
        # pt_data_path = os.path.join(self.args.data.base_dir, 'pre_training', 'tknzed_train_10pkts_per_flow.json')
        # pt_data = load_dataset('json', data_files=pt_data_path, split='train')
        
        
        # Reserve input_len for each packet <pkt> and <head> tokens. 
        # The last 1 is for <eos> token
        max_length = self.args.data.input_length + 2 * self.args.data.pkt_per_flow + 1

        optim_len, target_length = DataCollatorForPretrain.compute_input_and_target_lengths(
            inputs_length=self.args.data.input_length,
            noise_density=self.args.data.mlm_probability,
            mean_noise_span_length=self.args.data.mean_noise_span_length,
        )
        
        pt_data = pt_dataset(
            tokenizer=self.tokenizer,
            pt_data_path=os.path.join(self.args.data.base_dir, 'json/flow'),
            pkt_per_flow=self.args.data.pkt_per_flow,
            optim_len=optim_len,
            pop_percent=0.2,
            pop_switch_gap=5,
            nml_label_gap=4
        ).load_and_process()
                
        
        #TODO: Can be optimized within compute input and target lengths considering the special tokens that cannot be masked.
        self.logger.log_message(f"max_length = input_length+<pkt>+<head>+<eos>: {max_length}, expand_length based on input_length = {optim_len}, target_length = {target_length}")
        
        
                
        data_collator = DataCollatorForPretrain(
            max_length = max_length,
            optimal_len=optim_len,
            max_labels_length=target_length,
            noise_density = self.args.data.mlm_probability,
            mean_noise_span_length = self.args.data.mean_noise_span_length,
        )
        
        
        dataloaders = DataLoader(
            pt_data,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=self.args.optim.batch_size // self.args.optim.grad_acc,
            num_workers=self.args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        if self.args.optim.epochs > 0:
            assert not isinstance(pt_data, IterableDataset)    
            self.args.optim.total_steps = (len(dataloaders) // self.args.optim.grad_acc) * self.args.optim.epochs    
    
        self.train_dataloader = dataloaders
        #TODO[2024-0918]: Need overall test.
        
    def _get_lr_scheduler(self):
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.args, self.logger)
        
        
    def _prepare_compile(self):
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader
        )
        if self.args.model.compile:
            self.model = torch.compile(self.model)
    
    
    def _maybe_grad_clip_and_grad_calc(self):
        if self.args.optim.grad_clip > 0:
            grad_l2 = self.accelerator.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=self.args.optim.grad_clip,
                norm_type=2,
            )
        else:
            grad_l2 = None
            
        if self.args.logging.grad_l2:
            if grad_l2 is None:
                grad_l2 = (
                    sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5
                )

            self.stats.update({'grad_l2': grad_l2}) 


    def _record_update_loss_acc(self, output: dataclass):
        def safe_dict(obj):
            return {field.name: getattr(obj, field.name) for field in fields(obj)}
        
        for field, value in safe_dict(output).items():
            if isinstance(value, torch.Tensor):
                if field.endswith('loss') or field.endswith('acc'):
                    self.stats[field] = value.item()
            elif isinstance(value, (int, float)):
                self.stats[field] = value
                
        # Update averager
        self.train_averager.update(self.stats)
        
        
    def _maybe_logging(self):
        r"""
        Logger logging and accelerator logging
        """
        if self.current_train_step % self.args.logging.every_steps == 0:
            self.train_averager.update({'lr': self.optimizer.param_groups[0]['lr'], 
                                        'seconds_per_step': (time.time() - self.last_log_time) / self.args.logging.every_steps})
            averaged_stats = self.train_averager.average()
            
            for key, value in averaged_stats.items():
                if key.endswith('loss') or key.endswith('acc') or key.endswith('accuracy') or key.endswith('lr'):
                    if value is not None:
                        self.accelerator.log({key: value}, step=self.current_train_step)
         
            self.logger.log_stats(
                stats=averaged_stats,
                step=self.current_train_step,
                args=self.args,
                prefix='train/'
                )
            
            self.last_log_time = time.time()


    def _maybe_save_checkpoint(self):
        if (
            self.current_train_step >= self.args.optim.total_steps
            or self.current_train_step % self.args.checkpoint.every_steps == 0
        ):
            output_dir = f'checkpoint-{self.args.mode}-{self.current_train_step}'
            # adding safe_serialization=False to save shared parameters.
            self.accelerator.save_state(output_dir=output_dir, safe_serialization = False)    
    
    
    def train(self):
        self.model.train()
        while self.current_train_step <= self.args.optim.total_steps:        
            self.optimizer.zero_grad(set_to_none=True)
            for _, batch in enumerate(tqdm(self.train_dataloader, desc="Pre-training"), start=1):
                with self.accelerator.accumulate(self.model):
                    if self.current_train_step > self.args.optim.total_steps:
                        break
                    PretrainTasksOutput = self.model(**batch)
                    pt_loss_acc = self.__class__.calculate_loss_and_accuracy(PretrainTasksOutput)
                    self._record_update_loss_acc(pt_loss_acc)
                    loss = PretrainTasksOutput.msp_loss + 0.2*PretrainTasksOutput.pop_loss + 0.2*PretrainTasksOutput.nml_loss
                    # loss = PretrainTasksOutput.pop_loss + PretrainTasksOutput.nml_loss
                    # + PretrainTasksOutput.pop_loss 
                    # + PretrainTasksOutput.nml_loss
                    self.accelerator.backward(loss)
                    
                    self._maybe_grad_clip_and_grad_calc()
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True) 
                    
                    self._maybe_logging()
                    self._maybe_save_checkpoint()
                    self.current_train_step += 1
        self.accelerator.end_training()    
                
    def validate(self):
        """
        Using validation dataset to evaluate the model
        """
        self.model.eval()
        self.logger.log(f"=>Start validation on validation set")
        
    
                
if __name__ == '__main__':
    ...