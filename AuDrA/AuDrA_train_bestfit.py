"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022, December 2). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity. https://doi.org/10.31234/osf.io/t63dm

"""
from argparse import ArgumentParser
from AuDrA_DataModule import AuDrADataModule, RaterGeneralizationOneDataloader, RaterGeneralizationTwoDataloader, FarGeneralizationDataloader
from AuDrA_pl_plus_imgsums import AuDrA
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn, optim
import torchvision.models as models

#Set_seed
torch.manual_seed(42)




#  Hyperparams & Config
parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--architecture', default = 'resnet18')
parser.add_argument('--pretrained', default = True)
parser.add_argument('--in_shape', default = [3,224,224], type = int)
parser.add_argument('--img_means', default = 0.1610 , type = float)
parser.add_argument('--img_stds', default = 0.4072, type = float)
parser.add_argument('--num_outputs', default = 1, type = int)
parser.add_argument('--learning_rate', default = 0.00034664640432471026)
parser.add_argument('--batch_size', default = 16)
parser.add_argument('--train_pct', default = 0.7, type = float)
parser.add_argument('--val_pct', default = 0.1, type = float)
parser.add_argument('--loss_func', default = nn.MSELoss(), type = object)
parser.add_argument('--num_workers', default = 20)
args = parser.parse_args()


#  Init DataModules & Model
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='val_correlation',
    mode='max',
)

dm = AuDrADataModule(args = args)
dm.setup()
rg1_dataloader = RaterGeneralizationOneDataloader(args=args)
rg2_dataloader = RaterGeneralizationTwoDataloader(args=args)
fg_dataloader = FarGeneralizationDataloader(args=args)
model = AuDrA(args)


tb_logger = pl.loggers.TensorBoardLogger('logs/')
trainer = pl.Trainer.from_argparse_args(args,
                                        gpus = 1,
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
model.cur_filename = "rg1_output_dataframe.csv"
trainer.test(test_dataloaders=rg1_dataloader,verbose = False)
model.cur_filename = "rg2_output_dataframe.csv"
trainer.test(test_dataloaders=rg2_dataloader,verbose = False)
model.cur_filename = "fg_output_dataframe.csv"
trainer.test(test_dataloaders=fg_dataloader,verbose = False)
