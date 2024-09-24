import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import torch._dynamo
# torch._dynamo.config.optimize_ddp=False

from .src.pt_trainer import pt_trainer


@hydra.main(config_path="configs", config_name="debug_pt", version_base='1.1')
def main(args):
    pt_train = pt_trainer(args)
    pt_train.train()
    
if __name__ == "__main__":
    main()
    
    