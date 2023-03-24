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
    T5TokenizerFast as T5Tokenizer
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
N_EPOCHS = 8
BATCH_SIZE = 8
MODEL_NAME = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, max_length=512, truncation = True, padding='max_length')
    
class NewsSummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int = 512,
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
        tokenizer: T5Tokenizer,
        batch_size: int = 4,
        text_max_token_len: int = 512,
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

#text_token_counts, summary_token_counts = [], []

#for _, row in df_train_trimmed.iterrows():
#    text_token_count = len(tokenizer.encode(row['article']))
#    text_token_counts.append(text_token_count)

#    summary_token_count = len(tokenizer.encode(row['highlights']))
#    summary_token_counts.append(summary_token_count)  
#np.array(text_token_counts).dump(open('/content/drive/MyDrive/CNN DailyMail Summarisation Data/text_token_counts.npy', 'wb'))
#np.array(summary_token_counts).dump(open('/content/drive/MyDrive/CNN DailyMail Summarisation Data/summary_token_counts.npy', 'wb'))
#text_token_counts = np.load(open('/content/drive/MyDrive/CNN DailyMail Summarisation Data/text_token_counts.npy', 'rb'),allow_pickle=True)
#summary_token_counts = np.load(open('/content/drive/MyDrive/CNN DailyMail Summarisation Data/summary_token_counts.npy', 'rb'),allow_pickle=True)
#fig, (ax1, ax2) = plt.subplots(1, 2)
#sns.histplot(text_token_counts, ax=ax1)
#ax1.set_title('full text token counts')
#sns.histplot(summary_token_counts, ax=ax2)

class NewsSummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
    
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
            batch_size=batch_size
        )

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
            batch_size=batch_size
        )

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
            batch_size=batch_size
        )

        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)

def summarizeText(text):
    text_encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    generated_ids = trained_model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]
    return "".join(preds)

def main():
    sns.set(style='whitegrid', palette='muted', font_scale = 1.2)
    rcParams['figure.figsize'] = 16, 10

    pl.seed_everything(42)

    print("Is any cuda device available?",torch.cuda.is_available())
    print("Number of available cuda devices:",torch._C._cuda_getDeviceCount())

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
    
    model = NewsSummaryModel()

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

    print("path: ",trainer.checkpoint_callback.best_model_path,":::")

    trained_model = NewsSummaryModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    trained_model.freeze()
    
    sample_row = df_test_trimmed.iloc[0]
    text = sample_row['article']
    #print(text)

    model_summary = summarizeText(text)
    #print(model_summary)

    val_dataloaders = NewsSummaryDataModule(df_validation_trimmed, tokenizer)

    trainer.validate(model=model, dataloaders=val_dataloaders)

if __name__ == "__main__":
        main()
