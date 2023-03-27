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
import rouge
from nltk.translate.bleu_score import sentence_bleu
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_multiple_whitespaces, stem_text, strip_non_alphanum
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
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
BATCH_SIZE = 1
MODEL_NAME = 't5-small'
FT_MODEL_NAME = 'Alred/t5-small-finetuned-summarization-cnn'
tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_NAME, max_length=1024, truncation = True, padding='max_length')
    
class NewsSummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int = 1024,
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
        batch_size: int = 1,
        text_max_token_len: int = 1024,
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
            num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1
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
        self.model = T5ForConditionalGeneration.from_pretrained(FT_MODEL_NAME, return_dict=True)
    
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
            labels=labels
           # batch_size=batch_size
        )

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
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
            labels=labels
            #batch_size=batch_size
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
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
            labels=labels
            #batch_size=batch_size
        )

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)

def summarizeText(text):
    text_encoding = tokenizer(
        text,
        max_length=1024,
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

def get_rouge_and_bleu_scores (test_data):
    rouge = Rouge()
    ROUGE_SCORE_RUNNING_AVG = np.zeros((3, 3), dtype=float) #i -> R1 R2 R3 j -> f p r
    bleu_scores = np.zeroes((5), dtype = float) # 0 -> indiv 1-gram, 1 -> indiv 2-gram ... 3 -> indiv 4-gram, 4 -> cumul 4-gram
    count = 0
    for stuff in df_test_trimmed:
        count+=1
        text = stuff['article']
        model_summary = summarizeText(text)
        scores = rouge.get_scores(model_summary, stuff['highlights'])
        splitted_highlights = stuff['highlights'].split()
        splitted_inference = model_summary.split()
        bleu_scores[0] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (1,0,0,0)) - bleu_scores[0])/count
        bleu_scores[1] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0,1,0,0)) - bleu_scores[1])/count
        bleu_scores[2] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0,0,1,0)) - bleu_scores[2])/count
        bleu_scores[3] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0,0,0,1)) - bleu_scores[3])/count
        bleu_scores[4] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0.25,0.25,0.25,0.25)) - bleu_scores[4])/count
        ROUGE_SCORE_RUNNING_AVG[0][0] += (scores["rouge-1"]["f"] - ROUGE_SCORE_RUNNING_AVG[0][0])/count
        ROUGE_SCORE_RUNNING_AVG[0][1] += (scores["rouge-1"]["p"] - ROUGE_SCORE_RUNNING_AVG[0][1])/count
        ROUGE_SCORE_RUNNING_AVG[0][2] += (scores["rouge-1"]["r"] - ROUGE_SCORE_RUNNING_AVG[0][2])/count
        ROUGE_SCORE_RUNNING_AVG[1][0] += (scores["rouge-2"]["f"] - ROUGE_SCORE_RUNNING_AVG[1][0])/count
        ROUGE_SCORE_RUNNING_AVG[1][1] += (scores["rouge-2"]["p"] - ROUGE_SCORE_RUNNING_AVG[1][1])/count
        ROUGE_SCORE_RUNNING_AVG[1][2] += (scores["rouge-2"]["r"] - ROUGE_SCORE_RUNNING_AVG[1][2])/count
        ROUGE_SCORE_RUNNING_AVG[2][0] += (scores["rouge-l"]["f"] - ROUGE_SCORE_RUNNING_AVG[2][0])/count
        ROUGE_SCORE_RUNNING_AVG[2][1] += (scores["rouge-l"]["p"] - ROUGE_SCORE_RUNNING_AVG[2][1])/count
        ROUGE_SCORE_RUNNING_AVG[2][2] += (scores["rouge-l"]["r"] - ROUGE_SCORE_RUNNING_AVG[2][2])/count
    print("Rouge-1 Scores: f: {f1:4f}, p: {p1:4f}, r: {r1:4f}".format(f1 = ROUGE_SCORE_RUNNING_AVG[0][0], p1 = ROUGE_SCORE_RUNNING_AVG[0][1], r1 = ROUGE_SCORE_RUNNING_AVG[0][2]))
    print("Rouge-2 Scores: f: {f2:4f}, p: {p2:4f}, r: {r2:4f}".format(f2 = ROUGE_SCORE_RUNNING_AVG[1][0], p2 = ROUGE_SCORE_RUNNING_AVG[1][1], r2 = ROUGE_SCORE_RUNNING_AVG[1][2]))
    print("Rouge-L Scores: f: {f3:4f}, p: {p3:4f}, r: {r3:4f}".format(f3 = ROUGE_SCORE_RUNNING_AVG[2][0], p3 = ROUGE_SCORE_RUNNING_AVG[2][1], r3 = ROUGE_SCORE_RUNNING_AVG[2][2]))
    print("BLEU scores:: individual 1-gram : {b1:4f}, individual 2-gram : {b2:4f}, individual 3-gram : {b3:4f}, individual 4-gram : {b4:4f}, cumulative 4-gram : {b5:4f}".format(
    b1 = bleu_scores[0], b2 = bleu_scores[1], b3 = bleu_scores[2], b4 = bleu_scores[3], b5 = bleu_scores[4]))

def remove_stopwords_wrapper(df_test_trimmed, df_train_trimmed, df_validation_trimmed):
    print("starting stop word removal")
    for itr in range (0, len(df_test_trimmed)):
        stuff = df_test_trimmed['article'].iloc[itr]
        stuff =  remove_stopwords(stuff)
        df_test_trimmed['article'].iloc[itr] = stuff
        #print("removing stopwords test")
    print("done with stop word removal in test")
    for itr in range (0, len(df_train_trimmed)):
        stuff = df_train_trimmed['article'].iloc[itr]
        stuff =  remove_stopwords(stuff)
        df_train_trimmed['article'].iloc[itr] = stuff
        #print("removing stopwords train")
    print("done with stop word removal in train")
    for itr in range (0, len(df_validation_trimmed)):
        stuff = df_validation_trimmed['article'].iloc[itr]
        stuff =  remove_stopwords(stuff)
        df_validation_trimmed['article'].iloc[itr] = stuff
        #print("removing stopwords validation")
    print("done with stop word removal in validation")

def remove_stopwords_and_do_other_fancy_shmancy_stuff(df_test_trimmed, df_train_trimmed, df_validation_trimmed, stem):
    
    if stem:
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_multiple_whitespaces, remove_stopwords, stem_text]
    else:
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_multiple_whitespaces, remove_stopwords]

    for itr in range (0, len(df_test_trimmed)):
        stuff = df_test_trimmed['article'].iloc[itr]
        stuff = preprocess_string(stuff , CUSTOM_FILTERS)
        stuff = " ".join(stuff)
        df_test_trimmed['article'].iloc[itr] = stuff
    
    for itr in range (0, len(df_train_trimmed)):
        stuff = df_train_trimmed['article'].iloc[itr]
        stuff = preprocess_string(stuff , CUSTOM_FILTERS)
        stuff = " ".join(stuff)
        df_train_trimmed['article'].iloc[itr] = stuff
        
    for itr in range (0, len(df_validation_trimmed)):
        stuff = df_validation_trimmed['article'].iloc[itr]
        stuff = preprocess_string(stuff , CUSTOM_FILTERS)
        stuff = " ".join(stuff)
        df_validation_trimmed['article'].iloc[itr] = stuff
    
    
def main():

    print("wtf")

    pl.seed_everything(42)

    print("wtf 2")

    print("Is any cuda device available?",torch.cuda.is_available())
    print("Number of available cuda devices:",torch._C._cuda_getDeviceCount())

    test = "CNN DailyMail Summarisation Data/test.csv"
    train = "CNN DailyMail Summarisation Data/train.csv"
    validation = "CNN DailyMail Summarisation Data/validation.csv"

    df_train = pd.read_csv(train, encoding = "latin-1")
    df_test = pd.read_csv(test, encoding = "latin-1")
    df_validation = pd.read_csv(validation, encoding = "latin-1")

    df_train_trimmed = df_train[['article', 'highlights']]
    df_test_trimmed = df_test[['article', 'highlights']]
    df_validation_trimmed = df_validation[['article', 'highlights']]
    
    df_train_trimmed = df_train_trimmed[:100]
    df_test_trimmed = df_test_trimmed[:100]
    df_validation_trimmed = df_validation_trimmed[:100]
    
    
    
    remove_stopwords_wrapper(df_test_trimmed, df_train_trimmed, df_validation_trimmed)
    #remove_stopwords_and_do_other_fancy_shmancy_stuff(df_test_trimmed, df_train_trimmed, df_validation_trimmed, stem = True) #ALT POINT IN EXPERIMENT
    #remove_stopwords_and_do_other_fancy_shmancy_stuff(df_test_trimmed, df_train_trimmed, df_validation_trimmed, stem = False) #ALT POINT IN EXPERIMENT
    
    data_module = NewsSummaryDataModule(df_train_trimmed, df_test_trimmed, tokenizer = tokenizer)
    
    model = NewsSummaryModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath='/home/s2300928/AbstractionMLPProject/checkpoints',
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
        gpus = 2
    )

    trainer.fit(model, data_module)

    print("path: ",trainer.checkpoint_callback.best_model_path,":::")

    trained_model = NewsSummaryModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    trained_model.freeze()

    val_dataloaders = NewsSummaryDataModule(df_validation_trimmed, tokenizer = tokenizer)

    trainer.validate(model=model, dataloaders=val_dataloaders)

    get_rouge_and_bleu_scores(df_test_trimmed)
    
if __name__ == "__main__":
        main()
