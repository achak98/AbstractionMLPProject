# -*- coding: utf-8 -*-
"""AbstractiveSummarisation-T5small.ipynb

Automatically generated by Colaboratory.

"""

import json 
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

from tqdm.auto import tqdm

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
# %matplotlib inline
# %config InlineBackend.figure_format= 'retina'

torch.cuda.empty_cache()
N_EPOCHS = 1
BATCH_SIZE = 8

TOK_MAX_LENGTH = 512

MAX_LENGTH = 150
LENGTH_PENALTY = 1.0
EARLY_STOPPING = True
TEMPERATURE = 0.8
TOP_K = 40 
PENALTY_ALPHA = 0.6
REPETITION_PENALTY = 2.5

BASELINE_NAME = 't5-small'
MODEL_NAME = 'Alred/t5-small-finetuned-summarization-cnn'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, max_length=TOK_MAX_LENGTH, truncation = True, padding='max_length')
    
class NewsSummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        text_max_token_len: int = TOK_MAX_LENGTH,
        summary_max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row['article']

        text_encoding = tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        summary_encoding = tokenizer(
            data_row['highlights'],
            max_length=self.summary_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = summary_encoding['input_ids']
        labels[labels == 0] = -100 # to make sure we have correct labels for T5 text generation

        return dict(
            text=text,
            summary=data_row['highlights'],
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding['attention_mask'].flatten()
        )

class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        batch_size: int = 4,
        text_max_token_len: int = TOK_MAX_LENGTH,
        summary_max_token_len: int = 256
    ):
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):
        self.train_dataset = NewsSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.test_dataset = NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

class NewsSummaryModel(pl.LightningModule):
    def __init__(self, generate_kwargs : dict = {}, tokenizer_kwargs : dict = {}, do_int8 : bool = False, low_cpu_mem_usage : bool = False):
        super().__init__()
        self.generate_kwargs = generate_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, 
                                                                torch_dtype=torch.float16,
                                                                return_dict=True)
    
    def configure_optimizers(self, lr=1e-3):
        return AdamW(self.parameters(), lr=lr)

    def set_generator_kwargs(self, generator_kwargs):
        self.generate_kwargs = generator_kwargs
    
    def set_tokenizer_kwargs(self, tokenizer_kwargs):
        self.generate_kwargs = tokenizer_kwargs

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels, 
        )

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels)

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels)

        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        return loss
    
    def inference_step(self, text):
        text_encoding = tokenizer(text, **self.tokenizer_kwargs)
        generated_ids = self.model.generate(input_ids=text_encoding['input_ids'], attention_mask=text_encoding['attention_mask'], **self.generate_kwargs)

        return "".join([tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen_id in generated_ids])

def main():
    sns.set(style='whitegrid', palette='muted', font_scale = 1.2)
    rcParams['figure.figsize'] = 16, 10

    pl.seed_everything(42)

    print("Is any cuda device available?",torch.cuda.is_available())
    print("Number of available cuda devices:",torch._C._cuda_getDeviceCount())

    tokenizer_kwargs = {
        'max_length' : TOK_MAX_LENGTH,
        'padding' : 'max_length',
        'truncation' : True,
        'return_attention' : True,
        'add_special_tokens' : True,
        'return_tensors' : 'pt'
    }
    generate_kwargs = {
        'max_new_tokens':MAX_LENGTH,
        'temperature' : TEMPERATURE,
        'penalty_alpha' : PENALTY_ALPHA,
        'top_k' : TOP_K,
        'length_penalty' : LENGTH_PENALTY,
        'repetition_penalty' : REPETITION_PENALTY,
        'early_stopping' : EARLY_STOPPING
    }

    test = "CNN DailyMail Summarisation Data/test.csv"
    train = "CNN DailyMail Summarisation Data/train.csv"
    validation = "CNN DailyMail Summarisation Data/validation.csv"

    df_train = pd.read_csv(train, encoding = "latin-1")
    #df_train = df_train[500:]
    df_test = pd.read_csv(test, encoding = "latin-1")
    df_validation = pd.read_csv(validation, encoding = "latin-1")
    df_train.head()

    df_train_trimmed = df_train[['article', 'highlights']]
    df_test_trimmed = df_test[['article', 'highlights']]
    df_validation_trimmed = df_validation[['article', 'highlights']]
    df_train_trimmed.head()
    
    data_module = NewsSummaryDataModule(df_train_trimmed, df_test_trimmed, tokenizer)
    
    model = NewsSummaryModel(generate_kwargs=generate_kwargs, tokenizer_kwargs=tokenizer_kwargs, do_int8=True, low_cpu_mem_usage=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath='/checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    logger = TensorBoardLogger("lightning_logs", name='news-summary')
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=N_EPOCHS,
        gpus = 6
    )

    trainer.fit(model, data_module)

    print("path: ", trainer.checkpoint_callback.best_model_path,":::")

    trained_model = NewsSummaryModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    trained_model.freeze()
    
    sample_row = df_test_trimmed.iloc[0]
    text = sample_row['article']
    #print(text)

    model_summary = trained_model.inference_step(text)
    #print(model_summary)

    val_dataloaders = NewsSummaryDataModule(df_validation_trimmed, tokenizer)

    trainer.validate(model=model, dataloaders=val_dataloaders)

if __name__ == "__main__":
        main()
