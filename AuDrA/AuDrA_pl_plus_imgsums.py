"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022, December 2). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity. https://doi.org/10.31234/osf.io/t63dm

"""
from __future__ import print_function
from argparse import ArgumentParser
from collections import namedtuple, OrderedDict
import itertools
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from clip import clip
#from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer




_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x



class CustomCLIP_quality_all_metric_pro_diff_text(nn.Module):
    def __init__(self,share_parameter):
        super().__init__()
        self.share_parameter = share_parameter
        self.prompt_learner_1 = Prompt_learner_learnable_quality_singlemetric_pro_diff_text(share_parameter)
        self.prompt_learner_2 = Prompt_learner_learnable_quality_singlemetric_pro_diff_text(share_parameter)
        self.prompt_learner_3 = Prompt_learner_learnable_quality_singlemetric_pro_diff_text(share_parameter)
        self.prompt_learner_4 = Prompt_learner_learnable_quality_singlemetric_pro_diff_text(share_parameter)
        self.prompt_learner_5 = Prompt_learner_learnable_quality_singlemetric_pro_diff_text(share_parameter)

        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.tokenized_prompts_2 = self.prompt_learner_2.tokenized_prompts
        self.tokenized_prompts_3 = self.prompt_learner_3.tokenized_prompts
        self.tokenized_prompts_4 = self.prompt_learner_4.tokenized_prompts
        self.tokenized_prompts_5 = self.prompt_learner_5.tokenized_prompts


        self.image_encoder = share_parameter.clip_model.visual
        self.text_encoder = TextEncoder(share_parameter.clip_model)
        self.logit_scale = share_parameter.clip_model.logit_scale
        self.dtype = share_parameter.clip_model.dtype
        #self.factors = torch.tensor([1, 2, 3, 4, 5], dtype=self.dtype,requires_grad=False).to('cuda:0') #希望每个prompt_learner能明白自己在学习程度词
        #self.FC_layer = Super_FC()


    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        #image_features = self.image_encoder(image)

        prompts_1 = self.prompt_learner_1()
        prompts_2 = self.prompt_learner_2()
        prompts_3 = self.prompt_learner_3()
        prompts_4 = self.prompt_learner_4()
        prompts_5 = self.prompt_learner_5()

        tokenized_prompts_1 = self.tokenized_prompts_1
        tokenized_prompts_2 = self.tokenized_prompts_2
        tokenized_prompts_3 = self.tokenized_prompts_3
        tokenized_prompts_4 = self.tokenized_prompts_4
        tokenized_prompts_5 = self.tokenized_prompts_5

        text_features_1 = self.text_encoder(prompts_1, tokenized_prompts_1)
        text_features_2 = self.text_encoder(prompts_2, tokenized_prompts_2)
        text_features_3 = self.text_encoder(prompts_3, tokenized_prompts_3)
        text_features_4 = self.text_encoder(prompts_4, tokenized_prompts_4)
        text_features_5 = self.text_encoder(prompts_5, tokenized_prompts_5)

        text_features_1 = text_features_1 / text_features_1.norm(dim=-1, keepdim=True)
        text_features_2 = text_features_2 / text_features_2.norm(dim=-1, keepdim=True)
        text_features_3 = text_features_3 / text_features_3.norm(dim=-1, keepdim=True)
        text_features_4 = text_features_4 / text_features_4.norm(dim=-1, keepdim=True)
        text_features_5 = text_features_5 / text_features_5.norm(dim=-1, keepdim=True)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image_1 = logit_scale * image_features @ text_features_1.t()
        logits_per_image_2 = logit_scale * image_features @ text_features_2.t()
        logits_per_image_3 = logit_scale * image_features @ text_features_3.t()
        logits_per_image_4 = logit_scale * image_features @ text_features_4.t()
        logits_per_image_5 = logit_scale * image_features @ text_features_5.t()


        #logits_per_text = logits_per_image.t()

        return logits_per_image_1, logits_per_image_2, logits_per_image_3, logits_per_image_4, logits_per_image_5

    def do_batch(self,x):
        batch_size = x.size(0)
        num_patch = x.size(1)

        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        logits_per_image_1, logits_per_image_2, logits_per_image_3, logits_per_image_4, logits_per_image_5 = self.forward(x)

        logits_per_image_1 = logits_per_image_1.view(batch_size, num_patch, -1)
        logits_per_image_2 = logits_per_image_2.view(batch_size, num_patch, -1)
        logits_per_image_3 = logits_per_image_3.view(batch_size, num_patch, -1)
        logits_per_image_4 = logits_per_image_4.view(batch_size, num_patch, -1)
        logits_per_image_5 = logits_per_image_5.view(batch_size, num_patch, -1)

        logits_per_image_1 = logits_per_image_1.mean(1)
        logits_per_image_2 = logits_per_image_2.mean(1)
        logits_per_image_3 = logits_per_image_3.mean(1)
        logits_per_image_4 = logits_per_image_4.mean(1)
        logits_per_image_5 = logits_per_image_5.mean(1)

        logits_per_image_1 = F.softmax(logits_per_image_1, dim=1)
        logits_per_image_2 = F.softmax(logits_per_image_2, dim=1)
        logits_per_image_3 = F.softmax(logits_per_image_3, dim=1)
        logits_per_image_4 = F.softmax(logits_per_image_4, dim=1)
        logits_per_image_5 = F.softmax(logits_per_image_5, dim=1)

        #logits_per_image_1 = logits_per_image_1.view(-1, len(qualitys))  # 假设我们给设定了5组可学习的词向量
        #logits_per_image_2 = logits_per_image_2.view(-1, len(qualitys))
        #logits_per_image_3 = logits_per_image_3.view(-1, len(qualitys))
        #logits_per_image_4 = logits_per_image_4.view(-1, len(qualitys))
        #logits_per_image_5 = logits_per_image_5.view(-1, len(qualitys))  # batch*5

        # factors = torch.tensor([1, 2, 3, 4, 5], dtype=logits_per_image_1.dtype).to(device)

        #logits_quality_1 = logits_per_image_1 * self.factors
        #logits_quality_2 = logits_per_image_2 * self.factors
        #logits_quality_3 = logits_per_image_3 * self.factors  # 希望给5个评价维度的1*5的logit来个评价程度词的IQA的监督信号
        #logits_quality_4 = logits_per_image_4 * self.factors
        #logits_quality_5 = logits_per_image_5 * self.factors

        quality_preds_1 = 1 * logits_per_image_1[:, 0] + 2 * logits_per_image_1[:, 1] + 3 * logits_per_image_1[:, 2] + \
                          4 * logits_per_image_1[:, 3] + 5 * logits_per_image_1[:, 4]

        quality_preds_2 = 1 * logits_per_image_2[:, 0] + 2 * logits_per_image_2[:, 1] + 3 * logits_per_image_2[:, 2] + \
                          4 * logits_per_image_2[:, 3] + 5 * logits_per_image_2[:, 4]

        quality_preds_3 = 1 * logits_per_image_3[:, 0] + 2 * logits_per_image_3[:, 1] + 3 * logits_per_image_3[:, 2] + \
                          4 * logits_per_image_3[:, 3] + 5 * logits_per_image_3[:, 4]

        quality_preds_4 = 1 * logits_per_image_4[:, 0] + 2 * logits_per_image_4[:, 1] + 3 * logits_per_image_4[:, 2] + \
                          4 * logits_per_image_4[:, 3] + 5 * logits_per_image_4[:, 4]

        quality_preds_5 = 1 * logits_per_image_5[:, 0] + 2 * logits_per_image_5[:, 1] + 3 * logits_per_image_5[:, 2] + \
                          4 * logits_per_image_5[:, 3] + 5 * logits_per_image_5[:, 4]

        logits_quality = (quality_preds_1 + quality_preds_2 + quality_preds_3 + quality_preds_4 + quality_preds_5) / 5.0


        #print(f'logits_quality_1 = {logits_quality_1.shape}')
        #print(f'logits_quality_2 = {logits_quality_2.shape}')
        #print(f'logits_quality_3 = {logits_quality_3.shape}')
        #print(f'logits_quality_4 = {logits_quality_4.shape}')
        #print(f'logits_quality_5 = {logits_quality_5.shape}')
        #logits_quality = self.FC_layer(logits_quality_1, logits_quality_2, logits_quality_3, logits_quality_4,
        #                         logits_quality_5)


        return logits_quality
class Prompt_learner_learnable_quality_singlemetric_pro_diff_text(nn.Module):
    def __init__(self,share_parameter):
        super().__init__()
        self.share_parameter = share_parameter
        self.clip_model = share_parameter.clip_model
        self.dtype = share_parameter.clip_model.dtype
        self.learnable_quality = share_parameter.learnable_quality

        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",  # Metric and quality is learnable ，但是我们给程度词一些提示
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        #print(tokenized_prompts.shape)
        #print(tokenized_prompts[0])
        learnable_metric = torch.empty(1, 512, dtype=self.dtype)#每个这样的prompt_learner仅学习1个评价指标

        learnable_metric = torch.nn.init.normal_(learnable_metric, std=0.02)
        self.learnable_metric = nn.Parameter(learnable_metric)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :2, :])  # SOS
        self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])  # CLS, EOS
        self.tokenized_prompts = tokenized_prompts


    def forward(self):
        #embedding = self.embedding
        learnable_metric = self.learnable_metric
        if learnable_metric.dim() == 2:
            learnable_metric = learnable_metric.unsqueeze(0).expand(5, -1, -1)


        embedding_prefix = self.embedding_prefix
        embedding_postfix = self.embedding_postfix
        embedding_suffix = self.embedding_suffix

        #print(embedding_prefix.shape)
        #print(learnable_metric.shape)
        #print(embedding_postfix.shape)
        #print(self.learnable_quality.shape)
        #print(embedding_suffix.shape)

        embedding = torch.cat((embedding_prefix, learnable_metric, embedding_postfix,self.learnable_quality,embedding_suffix), dim=1
                            )  # 结果形状 [5, 77, 512]

        #print(f'Prompt_learner 返回的embedding形状 ：{embedding.shape}')
        return embedding

class Prompt_learner_learnable_quality(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype

        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",  # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        print(tokenized_prompts.shape)
        print(tokenized_prompts)

        learnable_quality = torch.empty(5, 512, dtype=self.dtype)

        learnable_quality = torch.nn.init.normal_(learnable_quality, std=0.02)
        self.learnable_quality = nn.Parameter(learnable_quality)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        #learnable_quality = embedding[:, 1:2, :]



        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])  # CLS, EOS
        self.tokenized_prompts = tokenized_prompts


    def forward(self):
        #embedding = self.embedding
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1)


        embedding_prefix = self.embedding_prefix
        #embedding_postfix = self.embedding_postfix
        embedding_suffix = self.embedding_suffix

        #print(embedding_prefix.shape)
        #print(learnable_quality.shape)
        #print(embedding_postfix.shape)
        #print(self.learnable_quality.shape)
        #print(embedding_suffix.shape)

        embedding = torch.cat((embedding_prefix, learnable_quality ,embedding_suffix), dim=1)  # 结果形状 [5, 77, 512]

        #print(f'Prompt_learner 返回的embedding形状 ：{embedding.shape}')
        return embedding

class Prompt_learner_learnable_quality_condition(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        print("RRRRRRRRRRRRRRRR")
        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])  # CLS, EOS
        self.tokenized_prompts = tokenized_prompts
        self.fc1 = nn.Linear(768, 768//16)
        self.fc2 = nn.Linear(768//16, 768)
        self.relu = nn.ReLU()

    def forward(self,image_features):
        condition_embedding = self.fc2(self.relu(self.fc1(image_features)))  # (BS,512)
        condition_embedding = condition_embedding.unsqueeze(1) #(BS,1,512)

        learnable_quality = self.learnable_quality.unsqueeze(0) #(1,5,512)
        print(learnable_quality.shape)
        shift_learnable_qualitys = learnable_quality + condition_embedding #(BS,5,512)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        prompts = []

        for shift in shift_learnable_qualitys:
            #shift:(5,512)
            shift_i = shift.unsqueeze(1)
            embedding = torch.cat((embedding_prefix,shift_i,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]
            #print(f'拼接后形状：{embedding.shape}')
            prompts.append(embedding)
        embeddings = torch.stack(prompts)

        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_condition(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_condition(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.softmax = nn.Softmax(dim=1)


    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        #print("XXXXXXXXXXX")
        #print(image_features.shape)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1(image_features) #(BS,5,77,768)

        logit_scale = self.logit_scale.exp()
        logits = []
        for prompt,image_feature in zip(prompts,image_features):
            text_features = self.text_encoder(prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_per_image = logit_scale * image_feature @ text_features.t()#(5,)
            logits.append(logits_per_image)
        logits = torch.stack(logits)

        logits  = self.softmax(logits)


        #直接算分数

        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]

        #logits_quality (BS,1)
        logits_quality.unsqueeze(1)


        return logits_quality

class CustomCLIP_quality_learnable_quality_FC(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        #self.share_parameter = share_parameter
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality(clip_model)

        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        #self.factors = torch.tensor([1, 2, 3, 4, 5], dtype=self.dtype,requires_grad=False).to('cuda:0') #希望每个prompt_learner能明白自己在学习程度词
        #self.FC_layer = Super_FC()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(5, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 512)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #image_features = self.image_encoder(image)
        #condition_embedding = self.fc2(self.relu(self.fc1(image_features))).mean(0)


        prompts_1 = self.prompt_learner_1()

        tokenized_prompts_1 = self.tokenized_prompts_1

        text_features_1 = self.text_encoder(prompts_1, tokenized_prompts_1)

        text_features_1 = text_features_1 / text_features_1.norm(dim=-1, keepdim=True)
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image_1 = logit_scale * image_features @ text_features_1.t()  #（BS，5）

        #直接算分数
        #print(logits_per_image_1.shape)

        logits_quality = self.relu(self.fc(logits_per_image_1))

        #logits_quality (BS,1)
        #print(logits_quality.shape)

        return logits_quality

class AuDrA_CLIP(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip.csv"
        ##CLIP
        clip_model, preprocess = clip.load("RN50", device='cpu', jit=False)

        #for p in clip_model.token_embedding.parameters():
        #    p.requires_grad = False
        #for p in clip_model.transformer.parameters():
        #    p.requires_grad = False
        #clip_model.positional_embedding.requires_grad = False
        #clip_model.text_projection.requires_grad = False
        #for p in clip_model.ln_final.parameters():
        #    p.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_rn50(clip_model,args)

    def forward(self, x):
        return self.models(x)

    def loss(self, pred_rating, true_rating):
        loss = self.args.loss_func(pred_rating, true_rating.view(-1,1))
        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'train_loss': pred_loss}

        return{'loss': pred_loss, 'log': log}


    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1), 'val_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs]) # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1) # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}


    def test_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()
        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1), 'test_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.args.learning_rate)



class AuDrA_CLIP_FC(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_fc.csv"

        ##CLIP
        clip_model, preprocess = clip.load("ViT-L/14", device='cpu', jit=False)
        clip_model.visual.requires_grad = False
        self.model = CustomCLIP_quality_learnable_quality_FC(clip_model)


    def forward(self, x):

        print(self.model(x))
        return self.model(x)

    def loss(self, pred_rating, true_rating):
        loss = self.args.loss_func(pred_rating, true_rating.view(-1,1))
        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'train_loss': pred_loss}

        return{'loss': pred_loss, 'log': log}


    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1), 'val_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs]) # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1) # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip_fc.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}


    def test_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()
        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1), 'test_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.args.learning_rate)


class AuDrA_CLIP_baseline(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_xiaorong_1.csv"

        clip_model, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
        clip_model.visual.requires_grad = False



        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_5_Q(clip_model,args)

        #print(self.prompts_baseline.device)


    def forward(self, x):

        return self.models(x)

    def loss(self, pred_rating, true_rating):
        loss = self.args.loss_func(pred_rating, true_rating.view(-1,1))
        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()
        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums:(16)
        print(f"sums :{sums.shape}")
        log = {'train_loss': pred_loss}

        return{'loss': pred_loss, 'log': log}


    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1), 'val_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs]) # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1) # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip_xiaorong_1.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}


    def test_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()
        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1), 'test_img_sums': sums,'test_fnames':fnames}

        return{'loss': pred_loss, 'log': log}


    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()

        fnames = []
        for x in outputs:
            batch_fnames = x['log']['test_fnames']
            # 如果 fnames 是张量，转换为列表
            if isinstance(batch_fnames, torch.Tensor):
                batch_fnames = batch_fnames.cpu().numpy().tolist()
            fnames.extend(batch_fnames)

        df = pd.DataFrame({'filename': fnames, 'ratings': ratings_np, 'predictions': preds_np})


        #df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.args.learning_rate)



class AuDrA(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_rn34.csv"

        # INITIALIZE MODEL
        try:
            self.model = models.__dict__[self.args.architecture](pretrained=self.args.pretrained)
        except:
            raise Exception('invalid architecture specified')

        # REMOVE LAST FULLY CONNECTED LAYER
        last_layer = list(self.model.children())[-1]
        if isinstance(last_layer, nn.Sequential):
            count = 0
            for layer in last_layer:
                if isinstance(layer, nn.Linear):
                    # fetch the first of the many Linear layers
                    count += 1
                    in_features = layer.in_features
                if count == 1:
                    break
        elif isinstance(last_layer, nn.Linear):
              in_features = last_layer.in_features

        # DEFINE NEW REGRESSION HEAD
        classifier = nn.Sequential(OrderedDict([
            ('bc1', nn.BatchNorm1d(in_features)),
            ('relu1', nn.ReLU()),
            ('fc1', nn.Linear(in_features, self.args.num_outputs, bias=True)),
        ]))

        # REPLACE THE CLASSIFIER WITH THE NEW REGRESSION HEAD
        if self.model.__dict__['_modules'].get('fc', None):
            self.model.fc = classifier
        else:
            self.model.classifier = classifier




    def forward(self, x):
        res = self.model(x)
        print(res)
        return res

    def loss(self, pred_rating, true_rating):
        loss = self.args.loss_func(pred_rating, true_rating.view(-1,1))
        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'train_loss': pred_loss}

        return{'loss': pred_loss, 'log': log}


    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1), 'val_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs]) # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1) # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_rn34.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}


    def test_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()
        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1), 'test_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.args.learning_rate)




class AuDrA_pytorch(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_pytorch.csv"

        # INITIALIZE MODEL
        try:
            self.model = models.__dict__[self.args.architecture](pretrained=self.args.pretrained)
        except:
            raise Exception('invalid architecture specified')

        # REMOVE LAST FULLY CONNECTED LAYER
        last_layer = list(self.model.children())[-1]
        if isinstance(last_layer, nn.Sequential):
            count = 0
            for layer in last_layer:
                if isinstance(layer, nn.Linear):
                    # fetch the first of the many Linear layers
                    count += 1
                    in_features = layer.in_features
                if count == 1:
                    break
        elif isinstance(last_layer, nn.Linear):
              in_features = last_layer.in_features

        # DEFINE NEW REGRESSION HEAD
        classifier = nn.Sequential(OrderedDict([
            ('bc1', nn.BatchNorm1d(in_features)),
            ('relu1', nn.ReLU()),
            ('fc1', nn.Linear(in_features, self.args.num_outputs, bias=True)),
        ]))

        # REPLACE THE CLASSIFIER WITH THE NEW REGRESSION HEAD
        if self.model.__dict__['_modules'].get('fc', None):
            self.model.fc = classifier
        else:
            self.model.classifier = classifier




    def forward(self, x):
        res = self.model(x)
        print(res)
        return res

    def loss(self, pred_rating, true_rating):
        loss = self.args.loss_func(pred_rating, true_rating.view(-1,1))
        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'train_loss': pred_loss}

        return{'loss': pred_loss, 'log': log}


    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1), 'val_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs]) # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1) # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_pytorch.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}


    def test_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()
        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1), 'test_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.args.learning_rate)





################################################################################



class Prompt_learner_learnable_quality_no_condition_VAR(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        self.prompt_style=[
            "a photo with abstract style",
            "a photo with concrete style",
            "a photo with hybrid abstract-concrete style",
        ]
        self.prompt_content=[
            "a photo of other",
            "a photo of object",
            "a photo of animal",
            "a photo of plant",
            "a photo of human"
        ]

        tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        self.register_buffer("prompts_style",embedding_style)
        self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_style = tokenized_prompt_style
        self.tokenized_prompts_content = tokenized_prompt_content

        self.fc1 = nn.Linear(512, 512//16)
        self.fc2 = nn.Linear(512//16, 768)
        self.relu = nn.ReLU()

    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_no_condition_VAR(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.softmax = nn.Softmax(dim=1)
        self.tokenized_prompts_styles = self.prompt_learner_1.tokenized_prompts_style
        #self.prompts_styles = self.prompt_learner_1.prompts_style

        #self.register_buffer("prompts_style", self.prompt_learner_1.prompts_style)


        self.tokenized_prompts_contents = self.prompt_learner_1.tokenized_prompts_content
        #self.prompts_contents = self.prompt_learner_1.prompts_content
        #self.register_buffer("prompts_content", self.prompt_learner_1.prompts_content)

        self.meta_styles = nn.Sequential(
            nn.Linear(3,1024//16),
            nn.ReLU(),
            nn.Linear(1024//16, 1024),
        )
        self.meta_contents = nn.Sequential(
            nn.Linear(5,1024//16),
            nn.ReLU(),
            nn.Linear(1024//16, 1024),
        )


    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        logit_scale = self.logit_scale.exp()

        #styles:(BS,)


        # (3,512)
        styles_features = self.text_encoder(self.prompt_learner_1.prompts_style, self.tokenized_prompts_styles)
        styles_features = styles_features / styles_features.norm(dim=-1, keepdim=True)
        logits_per_image_styles = logit_scale * image_features @ styles_features.t()  # (BS,3)
        logits_per_image_styles_softmax = self.softmax(logits_per_image_styles)

        styles_embeddings = self.meta_styles(logits_per_image_styles_softmax) #(BS,512)
        styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True) #归一化

        # (5,512)
        content_features = self.text_encoder( self.prompt_learner_1.prompts_content, self.tokenized_prompts_contents)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)

        logits_per_image = logit_scale * (image_features+contents_embeddings+styles_embeddings) @ text_features.t()  # (BS,5)


        logits  = self.softmax(logits_per_image)


        #直接算分数
        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)
        #print(logits_quality.shape)


        return logits_quality,logits_per_image_styles,logits_per_image_contents




class AuDrA_CLIP_baseline_VAR_1(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_1.csv"

        # clip_model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        # clip_model.visual.requires_grad = False

        ##CLIP
        clip_model, preprocess = clip.load("RN50", device='cpu', jit=False)

        #for p in clip_model.token_embedding.parameters():
        #    p.requires_grad = False
        #for p in clip_model.transformer.parameters():
        #    p.requires_grad = False
        #clip_model.positional_embedding.requires_grad = False
        #clip_model.text_projection.requires_grad = False
        #for p in clip_model.ln_final.parameters():
        #    p.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_1(clip_model)

        # print(self.prompts_baseline.device)

    def forward(self, x):

        return self.models(x) #quality,style,content

    def loss(self, pred_rating, true_rating,pre_style=None,true_style=None,pre_content=None,true_content=None):

        if pre_style is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

            loss2 = self.args.loss_func_class(pre_style, true_style)

            loss3 = self.args.loss_func_class(pre_content, true_content)

            loss = loss1 + 0.001 * (loss2 + loss3)
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch
        #style = style - 1
        #content = content - 1

        ratings = ratings.float()

        pred,style_logit,content_logit = self(images)
        pred_loss, mae = self.loss(pred, ratings,  style_logit,style,  content_logit,content )


        #images :(BS,3,224,224)
        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0

        log = {'train_loss': pred_loss}

        return {'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch

        #style = style - 1
        #content = content - 1
        ratings = ratings.float()

        pred,style_logit,content_logit = self(images)

        pred_loss, mae = self.loss(pred, ratings,style_logit,style,  content_logit,content)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0


        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1),
               'val_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs])  # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1)  # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip_1.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}

    def test_step(self, batch, batch_idx):
        #images, labels, ratings, fnames,style,content = batch

        #ratings = ratings.float()

        #pred = self(images)
        #pred_loss, mae = self.loss(pred, ratings)

        #pred, style_logit, content_logit = self(images)

        if self.cur_filename == "test_output_dataframe_clip_1.csv":
            images, labels, ratings, fnames, style, content = batch
            ratings = ratings.float()
            pred, style_logit, content_logit = self(images)
            pred_loss, mae = self.loss(pred, ratings, style_logit, style, content_logit, content)
        else:
            images, labels, ratings, fnames = batch
            ratings = ratings.float()
            pred, style_logit, content_logit = self(images)
            pred_loss, mae = self.loss(pred, ratings)


        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        #sums = 0.0

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1),
               'test_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)

class Prompt_learner_learnable_quality_no_condition_VAR_1(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        self.prompt_style=[
            "a photo with abstract style",
            "a photo with concrete style",
            "a photo with hybrid abstract-concrete style",
        ]
        self.prompt_content=[
            "a photo of other",
            "a photo of object",
            "a photo of animal",
            "a photo of plant",
            "a photo of human"
        ]

        tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        self.register_buffer("prompts_style",embedding_style)
        self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_style = tokenized_prompt_style
        self.tokenized_prompts_content = tokenized_prompt_content

        self.fc1 = nn.Linear(512, 512//16)
        self.fc2 = nn.Linear(512//16, 768)
        self.relu = nn.ReLU()

    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_1(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = PromptLearner_agiqa(["bad","poor","fair","good","perfect"],clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.softmax = nn.Softmax(dim=1)

        #self.prompts_styles = self.prompt_learner_1.prompts_style

        #self.register_buffer("prompts_style", self.prompt_learner_1.prompts_style)


        #self.tokenized_prompts_contents = self.prompt_learner_1.tokenized_prompts_content
        #self.prompts_contents = self.prompt_learner_1.prompts_content
        #self.register_buffer("prompts_content", self.prompt_learner_1.prompts_content)

        #self.meta_styles = nn.Sequential(
        #    nn.Linear(3,1024//16),
        #    nn.ReLU(),
        #    nn.Linear(1024//16, 1024),
        #)
        #self.meta_contents = nn.Sequential(
        #    nn.Linear(5,1024//16),
        #    nn.ReLU(),
        #    nn.Linear(1024//16, 1024),
        #)
        self.regression = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 6, 512),
            # nn.Linear(512, 512),  # only images
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 1),
        )



    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        logit_scale = self.logit_scale.exp()

        #styles:(BS,)


        # (3,512)
       # styles_features = self.text_encoder(self.prompt_learner_1.prompts_style, self.tokenized_prompts_styles)
       # styles_features = styles_features / styles_features.norm(dim=-1, keepdim=True)
       # logits_per_image_styles = logit_scale * image_features @ styles_features.t()  # (BS,3)
       # logits_per_image_styles_softmax = self.softmax(logits_per_image_styles)

       # styles_embeddings = self.meta_styles(logits_per_image_styles_softmax) #(BS,512)
       # styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True) #归一化

        # (5,512)
        #content_features = self.text_encoder( self.prompt_learner_1.prompts_content, self.tokenized_prompts_contents)
        #content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        #logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        #logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        #contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        #contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)


        #logits_per_image = logit_scale * (image_features) @ (text_features+contents_embeddings+styles_embeddings).t()  # (BS,5)


        logits = []
        image_features = image_features.unsqueeze(1)
        image_features = image_features.to(torch.float32)
        text_features = text_features.to(torch.float32)
        for image_feature in image_features:
            features = torch.cat((image_feature, text_features), dim=0)
            print(features.shape)
            features = features.flatten()
            # logit = self.fc1(features)
            logit = self.regression(features)
            logits.append(logit)
        logits = torch.stack(logits)
        logits_per_image = logits.squeeze(1)





        return logits_per_image



class AuDrA_CLIP_baseline_VAR_2(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_xiaorong_2.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device='cpu', jit=False)
        clip_model.visual.requires_grad = False

        ##CLIP
        #clip_model, preprocess = clip.load("RN50", device='cpu', jit=False)

        #for p in clip_model.token_embedding.parameters():
        #    p.requires_grad = False
        #for p in clip_model.transformer.parameters():
        #    p.requires_grad = False
        #clip_model.positional_embedding.requires_grad = False
        #clip_model.text_projection.requires_grad = False
        #for p in clip_model.ln_final.parameters():
        #    p.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_5_QD(clip_model,args)

        # print(self.prompts_baseline.device)

    def forward(self, x):

        return self.models(x) #quality,style,content

    def loss(self, pred_rating, true_rating,pre_content=None,true_content=None):

        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

            #loss2 = self.args.loss_func_class(pre_style, true_style)

            loss3 = self.args.loss_func_class(pre_content, true_content)

            loss = loss1 + 0.001 * (loss3)
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch
        #style = style - 1
        #content = content - 1

        ratings = ratings.float()

        pred,content_logit = self(images)
        pred_loss, mae = self.loss(pred, ratings,  content_logit,content)


        #images :(BS,3,224,224)
        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0

        log = {'train_loss': pred_loss}

        return {'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch

        #style = style - 1
        #content = content - 1
        ratings = ratings.float()

        pred,content_logit = self(images)

        pred_loss, mae = self.loss(pred, ratings,content_logit,content)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0


        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1),
               'val_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs])  # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1)  # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip_xiaorong_2.csv')



        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}

    def test_step(self, batch, batch_idx):
        #images, labels, ratings, fnames,style,content = batch
        #ratings = ratings.float()

        #pred = self(images)
        #pred_loss, mae = self.loss(pred, ratings)

        #pred,content_logit = self(images)

        if self.cur_filename == "test_output_dataframe_clip_xiaorong_2.csv":
            images, labels, ratings, fnames, style, content = batch
            ratings = ratings.float()
            pred, content_logit = self(images)
            pred_loss, mae = self.loss(pred, ratings, content_logit, content)
        else:
            images, labels, ratings, fnames = batch
            ratings = ratings.float()
            pred, _ = self(images)
            pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        #sums = 0.0

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1),
               'test_img_sums': sums,'test_fnames':fnames}

        return {'loss': pred_loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()

        fnames = []
        for x in outputs:
            batch_fnames = x['log']['test_fnames']
            # 如果 fnames 是张量，转换为列表
            if isinstance(batch_fnames, torch.Tensor):
                batch_fnames = batch_fnames.cpu().numpy().tolist()
            fnames.extend(batch_fnames)

        df = pd.DataFrame({'filename': fnames, 'ratings': ratings_np, 'predictions': preds_np})





        #df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)

class AuDrA_CLIP_baseline_VAR_22(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_xiaorong_2.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device='cpu', jit=False)

        #for p in clip_model.token_embedding.parameters():
        #    p.requires_grad = False
        #for p in clip_model.transformer.parameters():
        #    p.requires_grad = False
        #clip_model.positional_embedding.requires_grad = False
        #clip_model.text_projection.requires_grad = False
        #for p in clip_model.ln_final.parameters():
        #    p.requires_grad = False
        clip_model.visual.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_2(clip_model)

    def forward(self, x):
        return self.models(x)  # quality,style,content

    def loss(self, pred_rating, true_rating, pre_content=None, true_content=None):
        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))
            loss3 = self.args.loss_func_class(pre_content, true_content)
            loss = loss1 + 0.001 * (loss3)
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames, style, content = batch
        ratings = ratings.float()
        pred, content_logit = self(images)
        pred_loss, mae = self.loss(pred, ratings, content_logit, content)

        log = {'train_loss': pred_loss}
        return {'loss': pred_loss, 'log': log}

    def test_step(self, batch, batch_idx):
        if self.cur_filename == "test_output_dataframe_clip_xiaorong_2.csv":
            images, labels, ratings, fnames, style, content = batch
            ratings = ratings.float()
            pred, content_logit = self(images)
            pred_loss, mae = self.loss(pred, ratings, content_logit, content)
        else:
            images, labels, ratings, fnames = batch
            ratings = ratings.float()
            pred, _ = self(images)
            pred_loss, mae = self.loss(pred, ratings)

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1)}
        return {'loss': pred_loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu().numpy().tolist()
        preds_np = preds.cpu().numpy().tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch, 'test_correlation': correlation}
        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)



class Prompt_learner_learnable_quality_no_condition_VAR_2(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        self.prompt_style=[
            "a photo with abstract style",
            "a photo with concrete style",
            "a photo with hybrid abstract-concrete style",
        ]
        self.prompt_content=[
            "a photo of other",
            "a photo of object",
            "a photo of animal",
            "a photo of plant",
            "a photo of human"
        ]

        tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        self.register_buffer("prompts_style",embedding_style)
        self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_style = tokenized_prompt_style
        self.tokenized_prompts_content = tokenized_prompt_content

        self.fc1 = nn.Linear(512, 512//16)
        self.fc2 = nn.Linear(512//16, 768)
        self.relu = nn.ReLU()

    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_2(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_no_condition_VAR_2(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.softmax = nn.Softmax(dim=1)
        self.tokenized_prompts_styles = self.prompt_learner_1.tokenized_prompts_style
        #self.prompts_styles = self.prompt_learner_1.prompts_style

        #self.register_buffer("prompts_style", self.prompt_learner_1.prompts_style)


        self.tokenized_prompts_contents = self.prompt_learner_1.tokenized_prompts_content
        #self.prompts_contents = self.prompt_learner_1.prompts_content
        #self.register_buffer("prompts_content", self.prompt_learner_1.prompts_content)

        #self.meta_styles = nn.Sequential(
        #    nn.Linear(3,1024//16),
        #    nn.ReLU(),
        #    nn.Linear(1024//16, 1024),
        #)
        self.meta_contents = nn.Sequential(
            nn.Linear(5,768//16),
            nn.ReLU(),
            nn.Linear(768//16, 768),
        )


    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        logit_scale = self.logit_scale.exp()

        #styles:(BS,)


        # (3,512)
        #styles_features = self.text_encoder(self.prompt_learner_1.prompts_style, self.tokenized_prompts_styles)
        #styles_features = styles_features / styles_features.norm(dim=-1, keepdim=True)
        #logits_per_image_styles = logit_scale * image_features @ styles_features.t()  # (BS,3)
        #logits_per_image_styles_softmax = self.softmax(logits_per_image_styles)

        #styles_embeddings = self.meta_styles(logits_per_image_styles_softmax) #(BS,512)
        #styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True) #归一化

        # (5,512)
        content_features = self.text_encoder( self.prompt_learner_1.prompts_content, self.tokenized_prompts_contents)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)

        logits_per_image = logit_scale * (image_features+contents_embeddings) @ text_features.t()  # (BS,5)


        logits  = self.softmax(logits_per_image)


        #直接算分数
        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)
        #print(logits_quality.shape)


        return logits_quality,logits_per_image_contents



class AuDrA_CLIP_baseline_VAR_3(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_3.csv"

        # clip_model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        # clip_model.visual.requires_grad = False

        ##CLIP
        clip_model, preprocess = clip.load("RN50", device='cpu', jit=False)

        #for p in clip_model.token_embedding.parameters():
        #    p.requires_grad = False
        #for p in clip_model.transformer.parameters():
        #    p.requires_grad = False
        #clip_model.positional_embedding.requires_grad = False
        #clip_model.text_projection.requires_grad = False
        #for p in clip_model.ln_final.parameters():
        #    p.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_3(clip_model)

        # print(self.prompts_baseline.device)

    def forward(self, x):

        return self.models(x) #quality,style,content

    def loss(self, pred_rating, true_rating,pre_content=None,true_content=None):

        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

            #loss2 = self.args.loss_func_class(pre_style, true_style)

            loss3 = self.args.loss_func_class(pre_content, true_content)

            loss = loss1 + 0.001 * (loss3)
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch
        #style = style - 1
        #content = content - 1

        ratings = ratings.float()

        pred,content_logit = self(images)
        pred_loss, mae = self.loss(pred, ratings,  content_logit,content)


        #images :(BS,3,224,224)
        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0

        log = {'train_loss': pred_loss}

        return {'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch

        #style = style - 1
        #content = content - 1
        ratings = ratings.float()

        pred,content_logit = self(images)

        pred_loss, mae = self.loss(pred, ratings,content_logit,content)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0


        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1),
               'val_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs])  # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1)  # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip_3.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}

    def test_step(self, batch, batch_idx):
        #images, labels, ratings, fnames,style,content = batch

        #ratings = ratings.float()

        #pred = self(images)
        #pred_loss, mae = self.loss(pred, ratings)

        #pred,content_logit = self(images)

        if self.cur_filename == "test_output_dataframe_clip_3.csv":
            images, labels, ratings, fnames, style, content = batch
            ratings = ratings.float()
            pred, content_logit = self(images)
            pred_loss, mae = self.loss(pred, ratings, content_logit, content)
        else:
            images, labels, ratings, fnames = batch
            ratings = ratings.float()
            pred, _ = self(images)
            pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        #sums = 0.0

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1),
               'test_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)

class Prompt_learner_learnable_quality_no_condition_VAR_3(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        self.prompt_style=[
            "a photo with abstract style",
            "a photo with concrete style",
            "a photo with hybrid abstract-concrete style",
        ]
        self.prompt_content=[
            "a photo of other",
            "a photo of object",
            "a photo of animal",
            "a photo of plant",
            "a photo of human"
        ]

        tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        self.register_buffer("prompts_style",embedding_style)
        self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_style = tokenized_prompt_style
        self.tokenized_prompts_content = tokenized_prompt_content

        self.fc1 = nn.Linear(512, 512//16)
        self.fc2 = nn.Linear(512//16, 768)
        self.relu = nn.ReLU()

    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_3(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_no_condition_VAR_3(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.softmax = nn.Softmax(dim=1)
        self.tokenized_prompts_styles = self.prompt_learner_1.tokenized_prompts_style
        #self.prompts_styles = self.prompt_learner_1.prompts_style

        #self.register_buffer("prompts_style", self.prompt_learner_1.prompts_style)


        self.tokenized_prompts_contents = self.prompt_learner_1.tokenized_prompts_content
        #self.prompts_contents = self.prompt_learner_1.prompts_content
        #self.register_buffer("prompts_content", self.prompt_learner_1.prompts_content)

        #self.meta_styles = nn.Sequential(
        #    nn.Linear(3,1024//16),
        #    nn.ReLU(),
        #    nn.Linear(1024//16, 1024),
        #)
        self.meta_contents = nn.Sequential(
            nn.Linear(5,1024//16),
            nn.ReLU(),
            nn.Linear(1024//16, 1024),
        )


    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        logit_scale = self.logit_scale.exp()

        #styles:(BS,)


        # (3,512)
        #styles_features = self.text_encoder(self.prompt_learner_1.prompts_style, self.tokenized_prompts_styles)
        #styles_features = styles_features / styles_features.norm(dim=-1, keepdim=True)
        #logits_per_image_styles = logit_scale * image_features @ styles_features.t()  # (BS,3)
        #logits_per_image_styles_softmax = self.softmax(logits_per_image_styles)

        #styles_embeddings = self.meta_styles(logits_per_image_styles_softmax) #(BS,512)
        #styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True) #归一化

        # (5,512)
        content_features = self.text_encoder( self.prompt_learner_1.prompts_content, self.tokenized_prompts_contents)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)

        #logits_per_image = logit_scale * (image_features+contents_embeddings) @ text_features.t()  # (BS,5)


        #logits  = self.softmax(logits_per_image)

        logits = []
        for image_feature, contents_embedding in zip(image_features, contents_embeddings):
            image_feature = image_feature.unsqueeze(0)
            contents_embedding = contents_embedding.unsqueeze(0).expand(5, -1)
            conbined = contents_embedding + text_features

            logit = logit_scale * image_feature @ (conbined).t()
            # print(logit.shape)
            logits.append(logit)

        logits_per_image = torch.cat(logits, dim=0)

        # print(logits_per_image.shape)

        logits = self.softmax(logits_per_image)

        #直接算分数
        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)
        #print(logits_quality.shape)


        return logits_quality,logits_per_image_contents









class AuDrA_CLIP_baseline_VAR_4(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_4.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device='cpu', jit=False)
        clip_model.visual.requires_grad = False

        ##CLIP

        #clip_model, preprocess = clip.load("RN50", device='cpu', jit=False)
        #clip_model.visual.requires_grad = False
        #for p in clip_model.token_embedding.parameters():
        #    p.requires_grad = False
        #for p in clip_model.transformer.parameters():
        #    p.requires_grad = False
        #clip_model.positional_embedding.requires_grad = False
        #clip_model.text_projection.requires_grad = False
        #for p in clip_model.ln_final.parameters():
        #    p.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_5(clip_model,args)

        # print(self.prompts_baseline.device)

    def forward(self, x):

        return self.models(x) #quality,style,content

    def loss(self, pred_rating, true_rating,pre_content=None,true_content=None):

        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

            #loss2 = self.args.loss_func_class(pre_style, true_style)

            loss3 = self.args.loss_func_class(pre_content, true_content)

            loss = loss1 + 0.001 * (loss3)
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch
        #style = style - 1
        #content = content - 1

        ratings = ratings.float()
        pred,content_logit = self(images)
        pred_loss, mae = self.loss(pred, ratings,  content_logit,content)


        #images :(BS,3,224,224)
        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0

        log = {'train_loss': pred_loss}

        return {'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch

        #style = style - 1
        #content = content - 1
        ratings = ratings.float()

        pred,content_logit = self(images)

        pred_loss, mae = self.loss(pred, ratings,content_logit,content)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0


        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1),
               'val_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs])  # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1)  # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip_4.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}

    def test_step(self, batch, batch_idx):
        #images, labels, ratings, fnames,style,content = batch

        #ratings = ratings.float()

        #pred = self(images)
        #pred_loss, mae = self.loss(pred, ratings)

        #pred,content_logit = self(images)

        if self.cur_filename == "test_output_dataframe_clip_4.csv":
            images, labels, ratings, fnames, style, content = batch
            ratings = ratings.float()
            pred, content_logit = self(images)
            pred_loss, mae = self.loss(pred, ratings, content_logit, content)

        else:
            images, labels, ratings, fnames, = batch
            ratings = ratings.float()

            pred, _ = self(images)
            pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        #sums = 0.0

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1),
               'test_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)


class Prompt_learner_learnable_quality_no_condition_VAR_4(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        #self.prompt_style=[
        #    "a photo with abstract style",
        #    "a photo with concrete style",
        #    "a photo with hybrid abstract-concrete style",
        #]
        self.prompt_content=[
            "a photo of other",
            "a photo of object",
            "a photo of animal",
            "a photo of plant",
            "a photo of human"
        ]

        #tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            #embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        #self.register_buffer("prompts_style",embedding_style)
        self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        #self.tokenized_prompts_style = tokenized_prompt_style
        self.tokenized_prompts_content = tokenized_prompt_content

        #self.fc1 = nn.Linear(512, 512//16)
        #self.fc2 = nn.Linear(512//16, 768)
        #self.relu = nn.ReLU()

    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_4(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_no_condition_VAR_4(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.img_stds = args.img_stds
        self.img_means = args.img_means

        self.softmax = nn.Softmax(dim=1)
        #self.tokenized_prompts_styles = self.prompt_learner_1.tokenized_prompts_style
        #self.prompts_styles = self.prompt_learner_1.prompts_style

        #self.register_buffer("prompts_style", self.prompt_learner_1.prompts_style)


        self.tokenized_prompts_contents = self.prompt_learner_1.tokenized_prompts_content
        #self.prompts_contents = self.prompt_learner_1.prompts_content
        #self.register_buffer("prompts_content", self.prompt_learner_1.prompts_content)

        self.meta_ink = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(1,1024//16),
            nn.ReLU(),
            nn.Linear(1024//16, 1024),
        )

        self.meta_contents = nn.Sequential(
            nn.Linear(5,1024//16),
            nn.ReLU(),
            nn.Linear(1024//16, 1024),
        )


    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        unstandardized_imgs = [(img * self.img_stds + self.img_means) for img in image]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums:(BS,)
        sums = sums.unsqueeze(1)

        logit_scale = self.logit_scale.exp()

        #styles:(BS,)
        # (3,512)
        #styles_features = self.text_encoder(self.prompt_learner_1.prompts_style, self.tokenized_prompts_styles)
        #styles_features = styles_features / styles_features.norm(dim=-1, keepdim=True)
        #logits_per_image_styles = logit_scale * image_features @ styles_features.t()  # (BS,3)
        #logits_per_image_styles_softmax = self.softmax(logits_per_image_styles)

        #styles_embeddings = self.meta_styles(logits_per_image_styles_softmax) #(BS,512)
        #styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True) #归一化
        styles_embeddings = self.meta_ink(sums) #(BS,1024)
        styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True)  # 归一化




        # (5,512)
        content_features = self.text_encoder( self.prompt_learner_1.prompts_content, self.tokenized_prompts_contents)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)

        logits_per_image = logit_scale * (image_features+contents_embeddings+styles_embeddings) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)


        #直接算分数
        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)
        #print(logits_quality.shape)


        return logits_quality,logits_per_image_contents



class AuDrA_CLIP_baseline_VAR_5(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_CSCA.csv"
        self.training_test_result_filename = self.cur_filename
        self.training_val_result_filename = "validation_output_dataframe_clip_CSCA.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        clip_model.visual.requires_grad = False


        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_5(clip_model,args)



    def forward(self, x):

        return self.models(x)  #quality,content

    def loss(self, pred_rating, true_rating,pre_content=None,true_content=None):


        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

            loss2 = self.args.loss_func_class(pre_content, true_content)

            loss = loss1 + 0.001 * loss2
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,content = batch

        ratings = ratings.float()
        pred,content_logit = self(images)

        pred_loss, mae = self.loss(pred, ratings,  content_logit,content)


        #img :(BS,3,224,224)
        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])


        log = {'train_loss': pred_loss}

        return {'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames,content = batch


        ratings = ratings.float()

        pred,content_logit = self(images)

        pred_loss, mae = self.loss(pred, ratings,content_logit,content)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0


        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1),
               'val_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs])  # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1)  # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})

        df.to_csv(self.training_val_result_filename)

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}

    def test_step(self, batch, batch_idx):

        if self.cur_filename == self.training_test_result_filename:
            images, labels, ratings, fnames, content = batch
            ratings = ratings.float()
            pred, content_logit = self(images)
            pred_loss, mae = self.loss(pred, ratings, content_logit, content)

        else:
            images, labels, ratings, fnames, = batch
            ratings = ratings.float()

            pred, _ = self(images)
            pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        #sums = 0.0

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1),
               'test_img_sums': sums,'test_fnames':fnames}

        return {'loss': pred_loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()

        fnames = []
        for x in outputs:
            batch_fnames = x['log']['test_fnames']
            if isinstance(batch_fnames, torch.Tensor):
                batch_fnames = batch_fnames.cpu().numpy().tolist()
            fnames.extend(batch_fnames)

        df = pd.DataFrame({'filename': fnames,'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)






class AuDrA_CLIP_baseline_VAR_55(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_5.csv"

        ##CLIP
        clip_model, preprocess = clip.load("ViT-L/14", device='cpu', jit=False)
        clip_model.visual.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_5(clip_model,args)

    def forward(self, x):
        return self.models(x) #quality,style,content

    def loss(self, pred_rating, true_rating,pre_content=None,true_content=None):
        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))
            loss3 = self.args.loss_func_class(pre_content, true_content)
            loss = loss1 + 0.001 * (loss3)
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch
        ratings = ratings.float()
        pred,content_logit = self(images)
        pred_loss, mae = self.loss(pred, ratings,  content_logit,content)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'train_loss': pred_loss}
        return {'loss': pred_loss, 'log': log}

    def test_step(self, batch, batch_idx):
        images, labels, ratings, fnames, style, content = batch
        ratings = ratings.float()
        pred, content_logit = self(images)
        pred_loss, mae = self.loss(pred, ratings, content_logit, content)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1),
               'test_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)



class Prompt_learner_learnable_quality_no_condition_VAR_5(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype

        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]

        self.prompt_content=[
            "a photo of other",
            "a photo of object",
            "a photo of animal",
            "a photo of plant",
            "a photo of human"
        ]

        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)


        self.register_buffer("embedding_prefix", embedding[:, :7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_content = tokenized_prompt_content



    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1)  #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)    #(5, 77, 512)


        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_5(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = Prompt_learner_learnable_quality_no_condition_VAR_5(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.img_stds = args.img_stds
        self.img_means = args.img_means

        self.softmax = nn.Softmax(dim=1)



        self.tokenized_prompts_contents = self.prompt_learner.tokenized_prompts_content


        self.meta_ink = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(1,768//8),
            nn.ReLU(),
            nn.Linear(768//8, 768)
        )

        self.meta_contents = nn.Sequential(
            nn.Linear(5,768//16),
            nn.ReLU(),
            nn.Linear(768//16, 768)
        )


    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        unstandardized_imgs = [(img * self.img_stds + self.img_means) for img in image]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums:(BS,)
        sums = sums.unsqueeze(1)

        logit_scale = self.logit_scale.exp()


        styles_embeddings = self.meta_ink(sums) #(BS,768)
        styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True)  # 归一化




        # (5,512)
        content_features = self.text_encoder(self.prompt_learner.prompts_content, self.tokenized_prompts_contents)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)

        #logits = []
        #for image_feature, contents_embedding,styles_embedding in zip(image_features, contents_embeddings,styles_embeddings):
        #    image_feature = image_feature.unsqueeze(0)
        #    contents_embedding = contents_embedding.unsqueeze(0).expand(5, -1)
        #    styles_embedding = styles_embedding.unsqueeze(0).expand(5, -1)
        #    conbined = contents_embedding + text_features +  styles_embedding

        #    logit = logit_scale * image_feature @ (conbined).t()
            # print(logit.shape)
        #    logits.append(logit)

        #logits_per_image = torch.cat(logits, dim=0)

        logits_per_image = logit_scale * (image_features+contents_embeddings+styles_embeddings) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)


        #直接算分数
        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)
        #print(logits_quality.shape)


        return logits_quality,logits_per_image_contents


class Prompt_learner_learnable_quality_no_condition_VAR_5_QD(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        #self.prompt_style=[
        #    "a photo with abstract style",
        #    "a photo with concrete style",
        #    "a photo with hybrid abstract-concrete style",
        #]
        self.prompt_content=[
            "a photo of other",
            "a photo of object",
            "a photo of animal",
            "a photo of plant",
            "a photo of human"
        ]

        #tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            #embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        #self.register_buffer("prompts_style",embedding_style)
        self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        #self.tokenized_prompts_style = tokenized_prompt_style
        self.tokenized_prompts_content = tokenized_prompt_content

        #self.fc1 = nn.Linear(512, 512//16)
        #self.fc2 = nn.Linear(512//16, 768)
        #self.relu = nn.ReLU()

    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_5_QD(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_no_condition_VAR_5_QD(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.img_stds = args.img_stds
        self.img_means = args.img_means

        self.softmax = nn.Softmax(dim=1)
        #self.tokenized_prompts_styles = self.prompt_learner_1.tokenized_prompts_style
        #self.prompts_styles = self.prompt_learner_1.prompts_style

        #self.register_buffer("prompts_style", self.prompt_learner_1.prompts_style)


        self.tokenized_prompts_contents = self.prompt_learner_1.tokenized_prompts_content
        #self.prompts_contents = self.prompt_learner_1.prompts_content
        #self.register_buffer("prompts_content", self.prompt_learner_1.prompts_content)

        #self.meta_ink = nn.Sequential(
        #    nn.Sigmoid(),
        #    nn.Linear(1,768//8),
        #    nn.ReLU(),
        #    nn.Linear(768//8, 768)
        #)

        self.meta_contents = nn.Sequential(
            nn.Linear(5,768//16),
            nn.ReLU(),
            nn.Linear(768//16, 768)
        )


    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        #unstandardized_imgs = [(img * self.img_stds + self.img_means) for img in image]
        #sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums:(BS,)
        #sums = sums.unsqueeze(1)

        logit_scale = self.logit_scale.exp()

        #styles:(BS,)
        # (3,512)
        #styles_features = self.text_encoder(self.prompt_learner_1.prompts_style, self.tokenized_prompts_styles)
        #styles_features = styles_features / styles_features.norm(dim=-1, keepdim=True)
        #logits_per_image_styles = logit_scale * image_features @ styles_features.t()  # (BS,3)
        #logits_per_image_styles_softmax = self.softmax(logits_per_image_styles)

        #styles_embeddings = self.meta_styles(logits_per_image_styles_softmax) #(BS,512)
        #styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True) #归一化
        #styles_embeddings = self.meta_ink(sums) #(BS,1024)
        #styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True)  # 归一化




        # (5,512)
        content_features = self.text_encoder( self.prompt_learner_1.prompts_content, self.tokenized_prompts_contents)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)

        #logits = []
        #for image_feature, contents_embedding,styles_embedding in zip(image_features, contents_embeddings,styles_embeddings):
        #    image_feature = image_feature.unsqueeze(0)
        #    contents_embedding = contents_embedding.unsqueeze(0).expand(5, -1)
        #    styles_embedding = styles_embedding.unsqueeze(0).expand(5, -1)
        #    conbined = contents_embedding + text_features +  styles_embedding

        #    logit = logit_scale * image_feature @ (conbined).t()
            # print(logit.shape)
        #    logits.append(logit)

        #logits_per_image = torch.cat(logits, dim=0)

        logits_per_image = logit_scale * (image_features+contents_embeddings) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)


        #直接算分数
        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)
        #print(logits_quality.shape)


        return logits_quality,logits_per_image_contents


class Prompt_learner_learnable_quality_no_condition_VAR_5_QI(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]

        #self.prompt_content=[
        #    "a photo of other",
        #    "a photo of object",
        #    "a photo of animal",
        #    "a photo of plant",
        #    "a photo of human"
        #]

        #tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        #tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            #embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            #embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        #self.register_buffer("prompts_style",embedding_style)
        #self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        #self.tokenized_prompts_style = tokenized_prompt_style
        #self.tokenized_prompts_content = tokenized_prompt_content

    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_5_QI(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_no_condition_VAR_5_QI(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.img_stds = args.img_stds
        self.img_means = args.img_means

        self.softmax = nn.Softmax(dim=1)
        #self.tokenized_prompts_styles = self.prompt_learner_1.tokenized_prompts_style
        #self.prompts_styles = self.prompt_learner_1.prompts_style

        #self.register_buffer("prompts_style", self.prompt_learner_1.prompts_style)


        #self.tokenized_prompts_contents = self.prompt_learner_1.tokenized_prompts_content


        self.meta_ink = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(1,512//8),
            nn.ReLU(),
            nn.Linear(512//8, 512)
        )

        #self.meta_contents = nn.Sequential(
        #    nn.Linear(5,768//16),
        #    nn.ReLU(),
        #    nn.Linear(768//16, 768)
        #)


    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        unstandardized_imgs = [(img * self.img_stds + self.img_means) for img in image]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums:(BS,)
        sums = sums.unsqueeze(1)

        logit_scale = self.logit_scale.exp()

        #styles:(BS,)
        # (3,512)
        #styles_features = self.text_encoder(self.prompt_learner_1.prompts_style, self.tokenized_prompts_styles)
        #styles_features = styles_features / styles_features.norm(dim=-1, keepdim=True)
        #logits_per_image_styles = logit_scale * image_features @ styles_features.t()  # (BS,3)
        #logits_per_image_styles_softmax = self.softmax(logits_per_image_styles)

        #styles_embeddings = self.meta_styles(logits_per_image_styles_softmax) #(BS,512)
        #styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True) #归一化
        styles_embeddings = self.meta_ink(sums) #(BS,1024)
        styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True)  # 归一化




        # (5,512)
        #content_features = self.text_encoder( self.prompt_learner_1.prompts_content, self.tokenized_prompts_contents)
        #content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        #logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        #logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        #contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        #contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)

        logits_per_image = logit_scale * (image_features+styles_embeddings) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)


        #直接算分数
        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)
        #print(logits_quality.shape)


        return logits_quality


class Prompt_learner_learnable_quality_no_condition_VAR_5_Q(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        #self.prompt_style=[
        #    "a photo with abstract style",
        #    "a photo with concrete style",
        #    "a photo with hybrid abstract-concrete style",
        #]
        #self.prompt_content=[
        #    "a photo of other",
        #    "a photo of object",
        #    "a photo of animal",
        #    "a photo of plant",
        #    "a photo of human"
        #]

        #tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        #tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            #embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            #embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        #self.register_buffer("prompts_style",embedding_style)
        #self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        #self.tokenized_prompts_style = tokenized_prompt_style
        #self.tokenized_prompts_content = tokenized_prompt_content


    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_5_Q(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_no_condition_VAR_5_Q(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.img_stds = args.img_stds
        self.img_means = args.img_means

        self.softmax = nn.Softmax(dim=1)



    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        logit_scale = self.logit_scale.exp()



        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)



        logits_per_image = logit_scale * (image_features) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)



        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)



        return logits_quality





class AuDrA_CLIP_baseline_VAR_6(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_6.csv"

        # clip_model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        # clip_model.visual.requires_grad = False

        ##CLIP

        clip_model, preprocess = clip.load("ViT-L/14", device='cpu', jit=False)
        clip_model.visual.requires_grad = False
        #for p in clip_model.token_embedding.parameters():
        #    p.requires_grad = False
        #for p in clip_model.transformer.parameters():
        #    p.requires_grad = False
        #clip_model.positional_embedding.requires_grad = False
        #clip_model.text_projection.requires_grad = False
        #for p in clip_model.ln_final.parameters():
        #    p.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_6(clip_model,args)

        # print(self.prompts_baseline.device)

    def forward(self, x):

        return self.models(x) #quality,style,content

    def loss(self, pred_rating, true_rating,pre_content=None,true_content=None,pre_style=None, true_style=None):

        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

            loss2 = self.args.loss_func_class(pre_style, true_style)

            loss3 = self.args.loss_func_class(pre_content, true_content)

            loss = loss1 + 0.001 * (loss3+loss2)
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch
        #style = style - 1
        #content = content - 1

        ratings = ratings.float()
        pred,content_logit,style_logit = self(images)
        pred_loss, mae = self.loss(pred, ratings,  content_logit,content,style_logit,style)


        #images :(BS,3,224,224)
        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0

        log = {'train_loss': pred_loss}

        return {'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch

        #style = style - 1
        #content = content - 1
        ratings = ratings.float()

        pred,content_logit,style_logit = self(images)

        pred_loss, mae = self.loss(pred, ratings,content_logit,content,style_logit,style)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0


        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1),
               'val_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs])  # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1)  # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip_6.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}

    def test_step(self, batch, batch_idx):
        #images, labels, ratings, fnames,style,content = batch

        #ratings = ratings.float()

        #pred = self(images)
        #pred_loss, mae = self.loss(pred, ratings)

        #pred,content_logit = self(images)

        if self.cur_filename == "test_output_dataframe_clip_6.csv":
            images, labels, ratings, fnames, style, content = batch
            ratings = ratings.float()
            pred, content_logit,style_logit = self(images)
            pred_loss, mae = self.loss(pred, ratings, content_logit, content,style_logit,style)

        else:
            images, labels, ratings, fnames, = batch
            ratings = ratings.float()

            pred, _,_ = self(images)
            pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        #sums = 0.0

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1),
               'test_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)

class Prompt_learner_learnable_quality_no_condition_VAR_6(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        self.prompt_style=[
            "a photo with abstract style",
            "a photo with concrete style",
            "a photo with hybrid abstract-concrete style",
        ]
        self.prompt_content=[
            "a photo of other",
            "a photo of object",
            "a photo of animal",
            "a photo of plant",
            "a photo of human"
        ]

        tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        self.register_buffer("prompts_style",embedding_style)
        self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_style = tokenized_prompt_style
        self.tokenized_prompts_content = tokenized_prompt_content

        #self.fc1 = nn.Linear(512, 512//16)
        #self.fc2 = nn.Linear(512//16, 768)
        #self.relu = nn.ReLU()

    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_6(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_no_condition_VAR_6(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.img_stds = args.img_stds
        self.img_means = args.img_means

        self.softmax = nn.Softmax(dim=1)
        self.tokenized_prompts_styles = self.prompt_learner_1.tokenized_prompts_style
        self.prompts_styles = self.prompt_learner_1.prompts_style

        self.register_buffer("prompts_style", self.prompt_learner_1.prompts_style)


        self.tokenized_prompts_contents = self.prompt_learner_1.tokenized_prompts_content
        self.prompts_contents = self.prompt_learner_1.prompts_content
        self.register_buffer("prompts_content", self.prompt_learner_1.prompts_content)

        self.meta_ink = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(1,768//16),
            nn.ReLU(),
            nn.Linear(768//16, 768),
        )

        self.meta_contents = nn.Sequential(
            nn.Linear(5,768//16),
            nn.ReLU(),
            nn.Linear(768//16, 768),
        )
        self.meta_styles = nn.Sequential(
            nn.Linear(3, 768 // 16),
            nn.ReLU(),
            nn.Linear(768 // 16, 768),
        )



    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)

        unstandardized_imgs = [(img * self.img_stds + self.img_means) for img in image]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        #sums:(BS,)
        sums = sums.unsqueeze(1)

        logit_scale = self.logit_scale.exp()

        #styles:(BS,)
        # (3,512)
        styles_features = self.text_encoder(self.prompt_learner_1.prompts_style, self.tokenized_prompts_styles)
        styles_features = styles_features / styles_features.norm(dim=-1, keepdim=True)
        logits_per_image_styles = logit_scale * image_features @ styles_features.t()  # (BS,3)
        logits_per_image_styles_softmax = self.softmax(logits_per_image_styles)

        styles_embeddings = self.meta_styles(logits_per_image_styles_softmax) #(BS,512)
        styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True) #归一化


        ink_embeddings = self.meta_ink(sums)  # (BS,1024)
        ink_embeddings = ink_embeddings / ink_embeddings.norm(dim=-1, keepdim=True)  # 归一化



        # (5,512)
        content_features = self.text_encoder( self.prompt_learner_1.prompts_content, self.tokenized_prompts_contents)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)

        #logits = []
        #for image_feature, contents_embedding,styles_embedding in zip(image_features, contents_embeddings,styles_embeddings):
        #    image_feature = image_feature.unsqueeze(0)
        #    contents_embedding = contents_embedding.unsqueeze(0).expand(5, -1)
        #    styles_embedding = styles_embedding.unsqueeze(0).expand(5, -1)
        #    conbined = contents_embedding + text_features +  styles_embedding

        #    logit = logit_scale * image_feature @ (conbined).t()
        #    # print(logit.shape)
        #    logits.append(logit)

        #logits_per_image = torch.cat(logits, dim=0)

        logits_per_image = logit_scale * (image_features+contents_embeddings+styles_embeddings+ink_embeddings) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)


        #直接算分数
        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)
        #print(logits_quality.shape)


        return logits_quality,logits_per_image_contents,logits_per_image_styles





class Prompt_learner_learnable_quality_no_condition_VAR_7(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = self.clip_model.dtype
        #qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
        self.prompts = [
            "the creativity of the photo is bad",   # Quality is learnable
            "the creativity of the photo is poor",
            "the creativity of the photo is fair",
            "the creativity of the photo is good",
            "the creativity of the photo is perfect",
            ]
        #self.prompt_style=[
        #    "a photo with abstract style",
        #    "a photo with concrete style",
        #    "a photo with hybrid abstract-concrete style",
        #]
        self.prompt_content=[
            "a photo of other",
            "a photo of object",
            "a photo of animal",
            "a photo of plant",
            "a photo of human"
        ]

        #tokenized_prompt_style = torch.cat([clip.tokenize(text) for text in self.prompt_style])
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])
        tokenized_prompt_content = torch.cat([clip.tokenize(text) for text in self.prompt_content])

        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            #embedding_style = self.clip_model.token_embedding(tokenized_prompt_style).type(self.dtype)
            embedding_content = self.clip_model.token_embedding(tokenized_prompt_content).type(self.dtype)


        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)
        print(f'形状：{embedding.shape}')
        print(f'learnable_quality的形状：{learnable_quality.shape}')


        #self.embedding = embedding
        self.register_buffer("embedding_prefix", embedding[:, :7, :])  # SOS
        #self.register_buffer("embedding_postfix", embedding[:, 3:7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])
        #self.register_buffer("prompts_style",embedding_style)
        self.register_buffer("prompts_content", embedding_content)

        self.tokenized_prompts = tokenized_prompts
        #self.tokenized_prompts_style = tokenized_prompt_style
        self.tokenized_prompts_content = tokenized_prompt_content

        #self.fc1 = nn.Linear(512, 512//16)
        #self.fc2 = nn.Linear(512//16, 768)
        #self.relu = nn.ReLU()

    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1) #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)# 结果形状 [5, 77, 512]


        #print(f'Prompt_learner 返回的embeddings形状 ：{embeddings.shape}')
        return embeddings

class CustomCLIP_quality_learnable_quality_no_condition_VAR_7(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner_1 = Prompt_learner_learnable_quality_no_condition_VAR_7(clip_model)
        self.tokenized_prompts_1 = self.prompt_learner_1.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.softmax = nn.Softmax(dim=1)
        #self.tokenized_prompts_styles = self.prompt_learner_1.tokenized_prompts_style
        #self.prompts_styles = self.prompt_learner_1.prompts_style

        #self.register_buffer("prompts_style", self.prompt_learner_1.prompts_style)


        self.tokenized_prompts_contents = self.prompt_learner_1.tokenized_prompts_content
        self.prompts_contents = self.prompt_learner_1.prompts_content
        self.register_buffer("prompts_content", self.prompt_learner_1.prompts_content)

        #self.meta_styles = nn.Sequential(
        #    nn.Linear(3,1024//16),
        #    nn.ReLU(),
        #    nn.Linear(1024//16, 1024),
        #)
        self.meta_contents = nn.Sequential(
            nn.Linear(5,768//16),
            nn.ReLU(),
            nn.Linear(768//16, 768),
        )
        self.img_stds =   args.img_stds
        self.img_means =  args.img_means

        self.meta_ink = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(1, 768 // 16),
            nn.ReLU(),
            nn.Linear(768 // 16, 768),
        )


    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #print(image_features.shape)

        logit_scale = self.logit_scale.exp()

        unstandardized_imgs = [(img * self.img_stds + self.img_means) for img in image]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        # sums:(BS,)
        sums = sums.unsqueeze(1)

        ink_embeddings = self.meta_ink(sums)  # (BS,1024)
        ink_embeddings = ink_embeddings / ink_embeddings.norm(dim=-1, keepdim=True)  # 归一化



        #styles:(BS,)


        # (3,512)
        #styles_features = self.text_encoder(self.prompt_learner_1.prompts_style, self.tokenized_prompts_styles)
        #styles_features = styles_features / styles_features.norm(dim=-1, keepdim=True)
        #logits_per_image_styles = logit_scale * image_features @ styles_features.t()  # (BS,3)
        #logits_per_image_styles_softmax = self.softmax(logits_per_image_styles)

        #styles_embeddings = self.meta_styles(logits_per_image_styles_softmax) #(BS,512)
        #styles_embeddings = styles_embeddings / styles_embeddings.norm(dim=-1, keepdim=True) #归一化

        # (5,512)
        content_features = self.text_encoder(self.prompts_content, self.tokenized_prompts_contents)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        contents_embeddings = self.meta_contents(logits_per_image_contents_softmax) #(BS,512)
        contents_embeddings = contents_embeddings / contents_embeddings.norm(dim=-1, keepdim=True) #归一化


        tokenized_prompts = self.tokenized_prompts_1
        prompts = self.prompt_learner_1() #(BS,5,77,768)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape)
        #print(styles_embeddings.shape)
        #print(contents_embeddings.shape)

        logits_per_image = logit_scale * (image_features + contents_embeddings + ink_embeddings) @ text_features.t()  # (BS,5)


        logits  = self.softmax(logits_per_image)


        #直接算分数
        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        #logits_quality (BS,1)
        logits_quality =  logits_quality.unsqueeze(1)
        #print(logits_quality.shape)


        return logits_quality,logits_per_image_contents


class AuDrA_CLIP_baseline_VAR_7(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_7.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device='cpu', jit=False)
        clip_model.visual.requires_grad = False

        ##CLIP
        #clip_model, preprocess = clip.load("RN50", device='cpu', jit=False)

        #for p in clip_model.token_embedding.parameters():
        #    p.requires_grad = False
        #for p in clip_model.transformer.parameters():
        #    p.requires_grad = False
        #clip_model.positional_embedding.requires_grad = False
        #clip_model.text_projection.requires_grad = False
        #for p in clip_model.ln_final.parameters():
        #    p.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_7(clip_model,args)

        # print(self.prompts_baseline.device)

    def forward(self, x):

        return self.models(x) #quality,style,content

    def loss(self, pred_rating, true_rating,pre_content=None,true_content=None):

        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

            #loss2 = self.args.loss_func_class(pre_style, true_style)

            loss3 = self.args.loss_func_class(pre_content, true_content)

            loss = loss1 + 0.001 * (loss3)
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch
        #style = style - 1
        #content = content - 1

        ratings = ratings.float()

        pred,content_logit = self(images)
        pred_loss, mae = self.loss(pred, ratings,  content_logit,content)


        #images :(BS,3,224,224)
        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0

        log = {'train_loss': pred_loss}

        return {'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames,style,content = batch

        #style = style - 1
        #content = content - 1
        ratings = ratings.float()

        pred,content_logit = self(images)

        pred_loss, mae = self.loss(pred, ratings,content_logit,content)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums = 0.0


        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1),
               'val_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs])  # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1)  # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip_7.csv')



        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}

    def test_step(self, batch, batch_idx):
        #images, labels, ratings, fnames,style,content = batch
        #ratings = ratings.float()

        #pred = self(images)
        #pred_loss, mae = self.loss(pred, ratings)

        #pred,content_logit = self(images)

        if self.cur_filename == "test_output_dataframe_clip_7.csv":
            images, labels, ratings, fnames, style, content = batch
            ratings = ratings.float()
            pred, content_logit = self(images)
            pred_loss, mae = self.loss(pred, ratings, content_logit, content)
        else:
            images, labels, ratings, fnames = batch
            ratings = ratings.float()
            pred, _ = self(images)
            pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        #sums = 0.0

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1),
               'test_img_sums': sums}

        return {'loss': pred_loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)






class AuDrA_CLIP_only_LQ(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_xiaorong_3.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device='cpu', jit=False)
        clip_model.visual.requires_grad = False

        ##CLIP
        #clip_model, preprocess = clip.load("RN50", device='cpu', jit=False)
        #for p in clip_model.token_embedding.parameters():
        #    p.requires_grad = False
        #for p in clip_model.transformer.parameters():
        #    p.requires_grad = False
        #clip_model.positional_embedding.requires_grad = False
        #clip_model.text_projection.requires_grad = False
        #for p in clip_model.ln_final.parameters():
        #    p.requires_grad = False

        self.models = CustomCLIP_quality_learnable_quality_no_condition_VAR_5_Q(clip_model,args)

        #print(self.prompts_baseline.device)


    def forward(self, x):

        return self.models(x)

    def loss(self, pred_rating, true_rating):
        loss = self.args.loss_func(pred_rating, true_rating.view(-1,1))
        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()
        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])
        #sums:(16)
        print(f"sums :{sums.shape}")
        log = {'train_loss': pred_loss}

        return{'loss': pred_loss, 'log': log}


    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1), 'val_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs]) # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1) # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_output_dataframe_clip_xiaorong_3.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}


    def test_step(self, batch, batch_idx):
        images, labels, ratings, fnames = batch
        ratings = ratings.float()
        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1), 'test_img_sums': sums,'test_fnames':fnames}

        return{'loss': pred_loss, 'log': log}


    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("\n\n\n\n\nDataset: ", self.cur_filename)
        print("AuDrA-human correlation: ", str(correlation.item()))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs])
        sums = sums.unsqueeze(1)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Elaboration-human correlation: ", str(ink_correlation.item()))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()

        fnames = []
        for x in outputs:
            batch_fnames = x['log']['test_fnames']
            # 如果 fnames 是张量，转换为列表
            if isinstance(batch_fnames, torch.Tensor):
                batch_fnames = batch_fnames.cpu().numpy().tolist()
            fnames.extend(batch_fnames)

        df = pd.DataFrame({'filename': fnames, 'ratings': ratings_np, 'predictions': preds_np})


        #df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.args.learning_rate)





