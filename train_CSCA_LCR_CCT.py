from argparse import ArgumentParser
#from importlib.resources import contents

from AuDrA_DataModule import AuDrADataModule_CSCA, RaterGeneralizationOneDataloader, RaterGeneralizationTwoDataloader, FarGeneralizationDataloader
from CSCA_pl_plus_imgsums import CSCA_CLIP_LCR_CCT
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn, optim



#Set_seed
torch.manual_seed(42)

#  Hyperparams & Config
parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--in_shape', default = [3,224,224], type = int)
parser.add_argument('--img_means', default = 0.1610 , type = float)
parser.add_argument('--img_stds', default = 0.4072, type = float)

parser.add_argument('--learning_rate', default = 0.00001)
parser.add_argument('--batch_size', default = 16)
parser.add_argument('--train_pct', default = 0.7, type = float)
parser.add_argument('--val_pct', default = 0.1, type = float)
parser.add_argument('--loss_func', default = nn.MSELoss(), type = object)
parser.add_argument('--loss_func_class', default = nn.CrossEntropyLoss(), type = object)
parser.add_argument('--num_workers', default = 20)

#training datas and labels
parser.add_argument('--training_data_dir', default = '/home/cbl/IQA/zihao/AuDrA_Drawings/primary_images',   type = str)
parser.add_argument('--ratings_path', default = '/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/primary_jrt.csv',type = str)
parser.add_argument('--contents_path', default = '/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/1.csv',type = str)

#Testing datas and labels

parser.add_argument('--rg1_data_dir', default = '/home/cbl/IQA/zihao/AuDrA_Drawings/rater_generalization_one_images', type = str)
parser.add_argument('--rg1_ratings_path', default = '/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/rg1_jrt.csv',type = str)

parser.add_argument('--rg2_data_dir', default = '/home/cbl/IQA/zihao/AuDrA_Drawings/rater_generalization_two_images',   type = str)
parser.add_argument('--rg2_ratings_path', default = '/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/rg2_jrt.csv',type = str)

parser.add_argument('--fg_data_dir', default = '/home/cbl/IQA/zihao/AuDrA_Drawings/far_generalization_images',   type = str)
parser.add_argument('--fg_ratings_path', default = '/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/fg_jrt.csv',type = str)


args = parser.parse_args()


#  Init DataModules & Model
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='val_correlation',
    mode='max',
)

dm = AuDrADataModule_CSCA(args = args)
dm.setup()



#Testing dataloader
rg1_dataloader = RaterGeneralizationOneDataloader(args=args)
rg2_dataloader = RaterGeneralizationTwoDataloader(args=args)
fg_dataloader =  FarGeneralizationDataloader(args=args)

model = CSCA_CLIP_LCR_CCT(args)

tb_logger = pl.loggers.TensorBoardLogger('logs_clip_CSCA_LCR_CCT/')
trainer = pl.Trainer.from_argparse_args(args,
                                        gpus = [0],
                                        num_nodes = 1,
                                        deterministic = True,
                                        distributed_backend = 'dp',
                                        max_epochs = 136,
                                        precision = 16,
                                        logger = tb_logger,
                                        checkpoint_callback = checkpoint_callback,
)


# Train Model
trainer.fit(model, dm)

# Test Model
trainer.test(verbose = False)  # primary held-out dataset test
model.cur_filename = "rg1_output_dataframe_clip_CSCA_LCR_CCT.csv"
trainer.test(test_dataloaders=rg1_dataloader,verbose = False)
model.cur_filename = "rg2_output_dataframe_clip_CSCA_LCR_CCT.csv"
trainer.test(test_dataloaders=rg2_dataloader,verbose = False)
model.cur_filename = "fg_output_dataframe_clip_CSCA_LCR_CCT.csv"
trainer.test(test_dataloaders=fg_dataloader,verbose = False)
