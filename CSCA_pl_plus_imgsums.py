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


from clip import clip
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


class Prompt_learner(nn.Module):
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

class CustomCLIP(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = Prompt_learner(clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
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


        unstandardized_imgs = [(img * self.img_stds + self.img_means) for img in image]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])  #(BS,)

        sums = sums.unsqueeze(1)

        logit_scale = self.logit_scale.exp()


        PI_s = self.meta_ink(sums) #(BS,768)
        PI_s = PI_s / PI_s.norm(dim=-1, keepdim=True)


        content_features = self.text_encoder(self.prompt_learner.prompts_content, self.tokenized_prompts_contents)  # (5,768)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        PI_c = self.meta_contents(logits_per_image_contents_softmax) #(BS,768)
        PI_c = PI_c / PI_c.norm(dim=-1, keepdim=True)


        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)



        logits_per_image = logit_scale * (image_features + PI_c + PI_s) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)


        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        logits_quality =  logits_quality.unsqueeze(1) #(BS,1)



        return logits_quality,logits_per_image_contents

class CSCA_CLIP(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_CSCA.csv"
        self.training_test_result_filename = self.cur_filename
        self.training_val_result_filename = "validation_output_dataframe_clip_CSCA.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        clip_model.visual.requires_grad = False


        self.models = CustomCLIP(clip_model,args)



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




class Prompt_learner_LCR(nn.Module):
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



        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])


        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)



        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)


        self.register_buffer("embedding_prefix", embedding[:, :7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])


        self.tokenized_prompts = tokenized_prompts




    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1)  #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)    #(5, 77, 512)


        return embeddings

class CustomCLIP_LCR(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = Prompt_learner_LCR(clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
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




        logit_scale = self.logit_scale.exp()


        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)



        logits_per_image = logit_scale * (image_features) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)


        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        logits_quality =  logits_quality.unsqueeze(1) #(BS,1)



        return logits_quality

class CSCA_CLIP_LCR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_CSCA_LCR.csv"
        self.training_test_result_filename = self.cur_filename
        self.training_val_result_filename = "validation_output_dataframe_clip_CSCA_LCR.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        clip_model.visual.requires_grad = False


        self.models = CustomCLIP_LCR(clip_model,args)



    def forward(self, x):

        return self.models(x)  #quality,content

    def loss(self, pred_rating, true_rating,pre_content=None,true_content=None):


        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

            #loss2 = self.args.loss_func_class(pre_content, true_content)

            loss = loss1
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,content = batch

        ratings = ratings.float()
        pred = self(images)

        pred_loss, mae = self.loss(pred, ratings)


        #img :(BS,3,224,224)
        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])


        log = {'train_loss': pred_loss}

        return {'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames,content = batch


        ratings = ratings.float()

        pred = self(images)

        pred_loss, mae = self.loss(pred, ratings)

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
            pred= self(images)
            pred_loss, mae = self.loss(pred, ratings)

        else:
            images, labels, ratings, fnames, = batch
            ratings = ratings.float()

            pred= self(images)
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



class Prompt_learner_LCR_CCT(nn.Module):
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

class CustomCLIP_LCR_CCT(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = Prompt_learner_LCR_CCT(clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.img_stds = args.img_stds
        self.img_means = args.img_means

        self.softmax = nn.Softmax(dim=1)


        self.tokenized_prompts_contents = self.prompt_learner.tokenized_prompts_content


        self.meta_contents = nn.Sequential(
            nn.Linear(5,768//16),
            nn.ReLU(),
            nn.Linear(768//16, 768)
        )


    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)


        logit_scale = self.logit_scale.exp()


        content_features = self.text_encoder(self.prompt_learner.prompts_content, self.tokenized_prompts_contents)  # (5,768)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)

        logits_per_image_contents = logit_scale * image_features @ content_features.t()  # (BS,5)
        logits_per_image_contents_softmax = self.softmax(logits_per_image_contents)

        PI_c = self.meta_contents(logits_per_image_contents_softmax) #(BS,768)
        PI_c = PI_c / PI_c.norm(dim=-1, keepdim=True)


        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)



        logits_per_image = logit_scale * (image_features + PI_c) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)


        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        logits_quality =  logits_quality.unsqueeze(1) #(BS,1)



        return logits_quality,logits_per_image_contents

class CSCA_CLIP_LCR_CCT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_CSCA_LCR_CCT.csv"
        self.training_test_result_filename = self.cur_filename
        self.training_val_result_filename = "validation_output_dataframe_clip_CSCA_LCR_CCT.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        clip_model.visual.requires_grad = False


        self.models = CustomCLIP(clip_model,args)



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



class Prompt_learner_LCR_SCT(nn.Module):
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



        tokenized_prompts = torch.cat([clip.tokenize(text) for text in self.prompts])


        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)



        learnable_quality = embedding[:,7,:] #(5,768)
        self.learnable_quality = nn.Parameter(learnable_quality)


        self.register_buffer("embedding_prefix", embedding[:, :7, :])
        self.register_buffer("embedding_suffix", embedding[:, 8:, :])


        self.tokenized_prompts = tokenized_prompts




    def forward(self):
        learnable_quality = self.learnable_quality
        if learnable_quality.dim() == 2:
            learnable_quality = learnable_quality.unsqueeze(1)  #(5,1,768)

        embedding_prefix = self.embedding_prefix
        embedding_suffix = self.embedding_suffix

        embeddings = torch.cat((embedding_prefix,learnable_quality,embedding_suffix), dim=1)    #(5, 77, 512)


        return embeddings

class CustomCLIP_LCR_SCT(nn.Module):
    def __init__(self,clip_model,args):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = Prompt_learner_LCR_SCT(clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.img_stds = args.img_stds
        self.img_means = args.img_means

        self.softmax = nn.Softmax(dim=1)

        self.meta_ink = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(1, 768 // 8),
            nn.ReLU(),
            nn.Linear(768 // 8, 768)
        )




    def forward(self, image,styles=None,contents=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        unstandardized_imgs = [(img * self.img_stds + self.img_means) for img in image]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])  # (BS,)

        sums = sums.unsqueeze(1)

        logit_scale = self.logit_scale.exp()

        PI_s = self.meta_ink(sums)  # (BS,768)
        PI_s = PI_s / PI_s.norm(dim=-1, keepdim=True)





        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)



        logits_per_image = logit_scale * (image_features+PI_s) @ text_features.t()  # (BS,5)
        logits  = self.softmax(logits_per_image)


        logits_quality = 0.2 * logits[:, 0] + 0.4 * logits[:, 1] + 0.6 * logits[:, 2] + \
                         0.8 * logits[:, 3] + 1 * logits[:, 4]
        logits_quality =  logits_quality.unsqueeze(1) #(BS,1)



        return logits_quality

class CSCA_CLIP_LCR_SCT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cur_filename = "test_output_dataframe_clip_CSCA_LCR_SCT.csv"
        self.training_test_result_filename = self.cur_filename
        self.training_val_result_filename = "validation_output_dataframe_clip_CSCA_LCR_SCT.csv"

        clip_model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        clip_model.visual.requires_grad = False


        self.models = CustomCLIP_LCR_SCT(clip_model,args)



    def forward(self, x):

        return self.models(x)  #quality,content

    def loss(self, pred_rating, true_rating,pre_content=None,true_content=None):


        if pre_content is not None:
            loss1 = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

            #loss2 = self.args.loss_func_class(pre_content, true_content)

            loss = loss1
        else:
            loss = self.args.loss_func(pred_rating, true_rating.view(-1, 1))

        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings, fnames,content = batch

        ratings = ratings.float()
        pred = self(images)

        pred_loss, mae = self.loss(pred, ratings)


        #img :(BS,3,224,224)
        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])


        log = {'train_loss': pred_loss}

        return {'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings, fnames,content = batch


        ratings = ratings.float()

        pred = self(images)

        pred_loss, mae = self.loss(pred, ratings)

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
            pred= self(images)
            pred_loss, mae = self.loss(pred, ratings)

        else:
            images, labels, ratings, fnames, = batch
            ratings = ratings.float()

            pred= self(images)
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
