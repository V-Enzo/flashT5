import os
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from .collator import LongContextDataCollatorForT5MLM
from torch.utils.data import DataLoader
from omegaconf import open_dict
from .tool_utils import compute_input_and_target_lengths



def load_pretrain_dataloader(tokenizer, args):
    
    pre_training_dir = os.path.join(args.data.base_dir, 'pre_training','tknzed_train_10pkts_per_flow.json')
    print(f"=>Load PT dataset {pre_training_dir}")
    pt_train = load_dataset('json', data_files=pre_training_dir, split='train')
    pt_test = None
    
    
    before_mask_input_length, target_length = compute_input_and_target_lengths(
        inputs_length=args.data.input_length,
        noise_density=args.data.mlm_probability,
        mean_noise_span_length=args.data.mean_noise_span_length,
    )
    
    print("=>before_MSP_input_len: ", before_mask_input_length, "target_len ", target_length)
    data_collator = LongContextDataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=args.data.mlm_probability,
            mean_noise_span_length=args.data.mean_noise_span_length,
            input_length=args.data.input_length + 2*args.data.pkt_per_flow + 1,
            target_length=target_length,
            pkt_per_flow=args.data.pkt_per_flow,
            pad_token_id=0, # tokenizer.pad_token_id,
        )
    
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        pt_train,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.optim.batch_size // args.optim.grad_acc,
        num_workers=args.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # Add & Check args about data loaders
    is_iterable = isinstance(pt_train, IterableDataset)
    with open_dict(args):
        # if not is_iterable:
        #     args.data.train_batches = len(dataloaders['train'])
        #     if args.mode == 'ft':
        #         args.data.test_batches = len(dataloaders['test'])

        if args.optim.epochs > 0:
            assert not is_iterable
            args.optim.total_steps = (len(dataloaders['train']) // args.optim.grad_acc) * args.optim.epochs 

        # if hasattr(args.eval, 'eval_epochs') and args.eval.eval_epochs > 0:
        #     assert not is_iterable
        #     if args.mode == 'ft':
        #         args.eval.corrected_steps = (len(dataloaders['test']) // args.optim.grad_acc) * args.eval.eval_epochs
        #         print("args.eval.corrected_steps: ", args.eval.corrected_steps)
        # else:
        #     args.eval.corrected_steps = args.eval.steps
    

    return dataloaders['train'], None
