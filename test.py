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
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_multiple_whitespaces, stem_text, strip_non_alphanum
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    AutoTokenizer
)

from tqdm.auto import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
N_EPOCHS = 8
BATCH_SIZE = 2
NO_OF_WORKERS = 8
MODEL_NAME = 't5-small'
FT_MODEL_NAME = 'Alred/t5-small-finetuned-summarization-cnn'
tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_NAME, max_length=1024, truncation = True, padding='max_length')

class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int,
        text_max_token_len: int = 1024,
        summary_max_token_len: int = 256
    ):
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
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
        self.val_dataset = NewsSummaryDataset(
            self.val_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NO_OF_WORKERS,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NO_OF_WORKERS,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NO_OF_WORKERS,
            persistent_workers=True
        )

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
        self.batch_size = batch_size
        
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
        self.batch_size = batch_size
        
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
        self.batch_size = batch_size
        
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
        
def summarizeText(trained_model, text):
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

def get_rouge_and_bleu_scores (trained_model, df_test_trimmed):
    rouge = Rouge()
    ROUGE_SCORE_RUNNING_AVG = np.zeros((3, 3), dtype=float) #i -> R1 R2 R3 j -> f p r
    bleu_scores = np.zeros((5), dtype = float) # 0 -> indiv 1-gram, 1 -> indiv 2-gram ... 3 -> indiv 4-gram, 4 -> cumul 4-gram
    count = 0
    score_log1 = tqdm(total=0, position=1, bar_format='{desc}')
    score_log2 = tqdm(total=0, position=1, bar_format='{desc}')
    score_log3 = tqdm(total=0, position=1, bar_format='{desc}')
    score_log4 = tqdm(total=0, position=1, bar_format='{desc}')
    for itr in tqdm(range (0, len(df_test_trimmed)), desc = 'Processing Rouge and BLEU scores'):
        stuff = df_test_trimmed['article'].iloc[itr]
        what_stuffs_supposed_to_be = df_test_trimmed['highlights'].iloc[itr]
        count+=1
        model_summary = summarizeText(trained_model, stuff)
        rouge_scores = rouge.get_scores(model_summary, what_stuffs_supposed_to_be)
        splitted_highlights = what_stuffs_supposed_to_be.split()
        splitted_inference = model_summary.split()
        bleu_scores[0] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (1,0,0,0)) - bleu_scores[0])/count
        bleu_scores[1] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0,1,0,0)) - bleu_scores[1])/count
        bleu_scores[2] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0,0,1,0)) - bleu_scores[2])/count
        bleu_scores[3] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0,0,0,1)) - bleu_scores[3])/count
        bleu_scores[4] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0.25,0.25,0.25,0.25)) - bleu_scores[4])/count
       
        rouge_scores = rouge_scores[0]
        ROUGE_SCORE_RUNNING_AVG[0][0] += (rouge_scores["rouge-1"]["f"] - ROUGE_SCORE_RUNNING_AVG[0][0])/count
        ROUGE_SCORE_RUNNING_AVG[0][1] += (rouge_scores["rouge-1"]["p"] - ROUGE_SCORE_RUNNING_AVG[0][1])/count
        ROUGE_SCORE_RUNNING_AVG[0][2] += (rouge_scores["rouge-1"]["r"] - ROUGE_SCORE_RUNNING_AVG[0][2])/count
        ROUGE_SCORE_RUNNING_AVG[1][0] += (rouge_scores["rouge-2"]["f"] - ROUGE_SCORE_RUNNING_AVG[1][0])/count
        ROUGE_SCORE_RUNNING_AVG[1][1] += (rouge_scores["rouge-2"]["p"] - ROUGE_SCORE_RUNNING_AVG[1][1])/count
        ROUGE_SCORE_RUNNING_AVG[1][2] += (rouge_scores["rouge-2"]["r"] - ROUGE_SCORE_RUNNING_AVG[1][2])/count
        ROUGE_SCORE_RUNNING_AVG[2][0] += (rouge_scores["rouge-l"]["f"] - ROUGE_SCORE_RUNNING_AVG[2][0])/count
        ROUGE_SCORE_RUNNING_AVG[2][1] += (rouge_scores["rouge-l"]["p"] - ROUGE_SCORE_RUNNING_AVG[2][1])/count
        ROUGE_SCORE_RUNNING_AVG[2][2] += (rouge_scores["rouge-l"]["r"] - ROUGE_SCORE_RUNNING_AVG[2][2])/count
        
        score_log1.set_description_str("Rouge-1 Scores: f: {f1:4f}, p: {p1:4f}, r: {r1:4f}".format(f1 = ROUGE_SCORE_RUNNING_AVG[0][0], p1 = ROUGE_SCORE_RUNNING_AVG[0][1], r1 = ROUGE_SCORE_RUNNING_AVG[0][2]))
        score_log2.set_description_str("Rouge-2 Scores: f: {f2:4f}, p: {p2:4f}, r: {r2:4f}".format(f2 = ROUGE_SCORE_RUNNING_AVG[1][0], p2 = ROUGE_SCORE_RUNNING_AVG[1][1], r2 = ROUGE_SCORE_RUNNING_AVG[1][2]))
        score_log3.set_description_str("Rouge-L Scores: f: {f3:4f}, p: {p3:4f}, r: {r3:4f}".format(f3 = ROUGE_SCORE_RUNNING_AVG[2][0], p3 = ROUGE_SCORE_RUNNING_AVG[2][1], r3 = ROUGE_SCORE_RUNNING_AVG[2][2]))
        score_log4.set_description_str("BLEU scores:: individual 1-gram : {b1:4f}, individual 2-gram : {b2:4f}, individual 3-gram : {b3:4f}, individual 4-gram : {b4:4f}, cumulative 4-gram : {b5:4f}".format(
b1 = bleu_scores[0], b2 = bleu_scores[1], b3 = bleu_scores[2], b4 = bleu_scores[3], b5 = bleu_scores[4]))

def remove_stopwords_wrapper(df_test_trimmed, df_train_trimmed, df_validation_trimmed):
    print("starting stop word removal")
    for itr in tqdm(range (0, len(df_test_trimmed)), desc = 'Removing stopwords in test data'):
        stuff = df_test_trimmed['article'].iloc[itr]
        stuff =  remove_stopwords(stuff)
        df_test_trimmed['article'].iloc[itr] = stuff
    
    for itr in tqdm(range (0, len(df_train_trimmed)), desc = 'Removing stopwords in train data'):
        stuff = df_train_trimmed['article'].iloc[itr]
        stuff =  remove_stopwords(stuff)
        df_train_trimmed['article'].iloc[itr] = stuff
    
    for itr in tqdm(range (0, len(df_validation_trimmed)), desc = 'Removing stopwords in valdiation data'):
        stuff = df_validation_trimmed['article'].iloc[itr]
        stuff =  remove_stopwords(stuff)
        df_validation_trimmed['article'].iloc[itr] = stuff
        
    

def remove_stopwords_and_do_other_fancy_shmancy_stuff(df_test_trimmed, df_train_trimmed, df_validation_trimmed, stem):
    
    if stem:
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_multiple_whitespaces, remove_stopwords, stem_text]
    else:
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_multiple_whitespaces, remove_stopwords]

    for itr in tqdm(range (0, len(df_test_trimmed)), desc = 'Preprocessing test data'):
        stuff = df_test_trimmed['article'].iloc[itr]
        stuff = preprocess_string(stuff , CUSTOM_FILTERS)
        stuff = " ".join(stuff)
        df_test_trimmed['article'].iloc[itr] = stuff
    
    for itr in tqdm(range (0, len(df_train_trimmed)), desc = 'Preprocessing train data'):
        stuff = df_train_trimmed['article'].iloc[itr]
        stuff = preprocess_string(stuff , CUSTOM_FILTERS)
        stuff = " ".join(stuff)
        df_train_trimmed['article'].iloc[itr] = stuff
        
    for itr in tqdm(range (0, len(df_validation_trimmed)), desc = 'Preprocessing valdiation data'):
        stuff = df_validation_trimmed['article'].iloc[itr]
        stuff = preprocess_string(stuff , CUSTOM_FILTERS)
        stuff = " ".join(stuff)
        df_validation_trimmed['article'].iloc[itr] = stuff

def main():

    pl.seed_everything(42)

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
    
    #remove_stopwords_wrapper(df_test_trimmed, df_train_trimmed, df_validation_trimmed)
    #remove_stopwords_and_do_other_fancy_shmancy_stuff(df_test_trimmed, df_train_trimmed, df_validation_trimmed, stem = True) #ALT POINT IN EXPERIMENT
    #remove_stopwords_and_do_other_fancy_shmancy_stuff(df_test_trimmed, df_train_trimmed, df_validation_trimmed, stem = False) #ALT POINT IN EXPERIMENT
    
    data_module = NewsSummaryDataModule(df_train_trimmed, df_test_trimmed, df_validation_trimmed, tokenizer = tokenizer, batch_size = BATCH_SIZE)

    trained_model = NewsSummaryModel.load_from_checkpoint(
        checkpoint_path="/home/s2300928/AbstractionMLPProject/checkpoints/best-checkpoint.ckpt"
    )
    trained_model.freeze()

    get_rouge_and_bleu_scores(trained_model, df_test_trimmed)
    
if __name__ == "__main__":
        main()

