import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_multiple_whitespaces, stem_text, strip_non_alphanum
import transformers
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    AutoTokenizer,
    Pipeline
)

from tqdm.auto import tqdm
import evaluate

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
N_EPOCHS = 8
BATCH_SIZE = 1
NO_OF_WORKERS = 0
MODEL_NAME = 't5-small'
FT_MODEL_NAME = 'Alred/t5-small-finetuned-summarization-cnn'
tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_NAME, max_length=1024, truncation = True, padding='max_length')

generating_heatmap = False

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
            persistent_workers=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NO_OF_WORKERS,
            persistent_workers=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NO_OF_WORKERS,
            persistent_workers=False
        )
        
class NewsSummaryDataModuleTest(pl.LightningDataModule):
    def __init__(
        self,
        
        test_df: pd.DataFrame,
        
        tokenizer: T5Tokenizer,
        batch_size: int,
        text_max_token_len: int = 1024,
        summary_max_token_len: int = 256
    ):
        super().__init__()

        
        self.test_df = test_df
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):
        
        self.test_dataset = NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NO_OF_WORKERS,
            persistent_workers=False
        )


class NewsSummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(FT_MODEL_NAME, output_attentions = True, return_dict_in_generate=True)
    
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        
        text = "Automatic text summarisation aims to produce a brief but comprehensive version of one or multiple documents, highlighting the most important information. There are two main summarisation techniques: extractive and abstractive. Extractive summarisation involves selecting key sentences from the original document, while abstractive summarisation involves creating new language based on the important information and requires a deeper understanding of the content."
        input_ids = tokenizer.encode(text, return_tensors='pt')
        outputs = trained_model.model.generate(input_ids=input_ids, max_length=100, num_beams=4, early_stopping=True)
        model_summary = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)

        text_input_ids = tokenizer.encode_plus(text, return_tensors='pt')['input_ids']
    
        summary_input_ids = tokenizer.encode_plus(model_summary, return_tensors='pt')['input_ids']
        last_layer_attention_cross = output['cross_attentions'][-1]
        last_layer_attention_enc = output['encoder_attentions'][-1]
        last_layer_attention_dec = output['decoder_attentions'][-1]
        summary_attention_cross = last_layer_attention_cross[:, :, -len(summary_input_ids[0]):, :]
        summary_attention_enc = last_layer_attention_enc[:, :, -len(summary_input_ids[0]):, :]
        summary_attention_dec = last_layer_attention_dec[:, :, -len(summary_input_ids[0]):, :]

        # Sum the attention scores across the heads and normalize them
        summary_attention_cross = summary_attention_cross.sum(dim=1, keepdim =True)
        summary_attention_cross /= summary_attention_cross.sum(dim=-1, keepdim=True)
# Sum the attention scores across the heads and normalize them
        summary_attention_enc = summary_attention_enc.sum(dim=1, keepdim =True)
        summary_attention_enc /= summary_attention_enc.sum(dim=-1, keepdim=True)
# Sum the attention scores across the heads and normalize them
        summary_attention_dec = summary_attention_dec.sum(dim=1, keepdim =True)
        summary_attention_dec /= summary_attention_dec.sum(dim=-1, keepdim=True)

        # Convert the attention scores to a numpy array
        summary_attention_cross = summary_attention_cross.detach().cpu().numpy()
        summary_attention_enc = summary_attention_enc.detach().cpu().numpy()
        summary_attention_dec = summary_attention_dec.detach().cpu().numpy()
        

        sns.set(style='whitegrid', font_scale=1)
        rcParams['figure.figsize'] = 80, 40
        rc('font')
        summary_attention_cross = summary_attention_cross.squeeze(0)
        summary_attention_enc = summary_attention_enc.squeeze(0)
        summary_attention_dec = summary_attention_dec.squeeze(0)
        x = [tokenizer.decode(token) for token in text_input_ids[0]]
        y = [tokenizer.decode(token) for token in summary_input_ids[0]]
        
        sns.set(font_scale=2.1)
        ax = sns.heatmap(summary_attention_cross[0], cmap='Spectral_r', annot=True, fmt='.1f', cbar=False)
        
        ax.set_xticklabels(x, rotation=90, fontsize=40)
        ax.set_yticklabels(y, rotation=0, fontsize=40)
        ax.set_xticks(np.arange(len(x))+0.5)
        ax.set_yticks(np.arange(len(y))+0.5)
        #ax.set_yticklabels([''])
        ax.set_xlabel('Input Tokens', fontsize=60, fontweight='bold')
        ax.set_ylabel('Output Tokens', fontsize=60, fontweight='bold')
        ax.set_xlabel('Output Tokens', fontsize=60, fontweight='bold')
        ax.set_ylabel('Input Tokens', fontsize=60, fontweight='bold')
        #ax.set_title('Attention Heatmap', fontsize=40, fontweight='bold')
        plt.savefig('baseline/heatmap_cross.pdf', format='pdf', dpi=300, bbox_inches='tight')
        ax = sns.heatmap(summary_attention_enc[0], cmap='Spectral_r', annot=True, fmt='.1f', cbar=False)
        plt.savefig('baseline/heatmap_enc.pdf', format='pdf', dpi=300, bbox_inches='tight')
        ax = sns.heatmap(summary_attention_dec[0], cmap='Spectral_r', annot=True, fmt='.1f', cbar=False)
        # Save the plot in a pdf file
        plt.savefig('baseline/heatmap_dec.pdf', format='pdf', dpi=300, bbox_inches='tight')


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

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
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

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
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
    
        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def predict_step(self, batch, batch_size):
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
        generated_ids = self.model.generate(
            input_ids=batch['text_input_ids'],
            attention_mask=batch['text_attention_mask'],
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = []
        for i in range(len(generated_ids['sequences'])):
            genid = generated_ids['sequences'][i]
        return tokenizer.decode(genid, skip_special_tokens=True)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)
    
    def generate_attention_map(self, text_input_ids, summary_input_ids, text, model_summary):
        # Set the model to evaluation mode
        self.model.eval()

        # Generate the output
        output = self.model.generate(
	        input_ids=text_input_ids,
	        attention_mask=(text_input_ids != tokenizer.pad_token_id),
	    )
        print("output's class: ", type(output).__name__)
        print("input sequences length: ",len(text_input_ids[0]))
        print("output sequences length: ",len(summary_input_ids[0]))
        summary_attention_average = np.zeros((1, 1, len(summary_input_ids[0]),len(text_input_ids[0])))
        print("no of gen tokens: ", len(output['cross_attentions']))
        print("no of decoder layers: ", len(output['cross_attentions'][0]))
        count = 0
        for tuple_gen_token in tqdm(output['cross_attentions'], desc = 'Processing Attention heatmap'):
            count+=1
            print("float tensor shape for iteration {} : {}".format(count, (tuple_gen_token[-1].size())))
            # Get the attention scores from the last layer of the decoder
            last_layer_attention = tuple_gen_token[-1] #(batch_size, num_heads, generated_length, sequence_length).
            
            # Reshape the attention scores to match the output shape
            #last_layer_attention = torch.stack(list(last_layer_attention), dim=0)
            #last_layer_attention = last_layer_attention.squeeze(0)
            #last_layer_attention = last_layer_attention.view(
        #        output['sequences'].size(0),
        #        self.model.config.num_heads,
        #        -1,
        #        output['sequences'].size(-1)
        #    )
            # Compute the attention scores for the summary tokens
            summary_attention = last_layer_attention[:, :, -len(summary_input_ids[0]):, :]

            # Sum the attention scores across the heads and normalize them
            summary_attention = summary_attention.sum(dim=1, keepdim =True)
            summary_attention /= summary_attention.sum(dim=-1, keepdim=True)
        
            # Convert the attention scores to a numpy array
            summary_attention = summary_attention.detach().cpu().numpy()
            
            for j in range (len(summary_attention[0][0][0])):
                summary_attention_average[0][0][count][j] += summary_attention[0][0][0][j]
            
            
            
	    # Plot the heatmap
        sns.set(style='whitegrid', font_scale=1)
        rcParams['figure.figsize'] = 80, 40
        rc('font')
        summary_attention_average = summary_attention_average.squeeze(0)
        y = [tokenizer.decode(token) for token in summary_input_ids[0]]
        x = [tokenizer.decode(token) for token in text_input_ids[0]]
        sns.set(font_scale=2.1)
        ax = sns.heatmap(summary_attention_average[0], cmap='Spectral_r', annot=True, fmt='.1f', cbar=False)
        ax.set_xticklabels(x, rotation=90, fontsize=40)
        ax.set_yticklabels(y, rotation=0, fontsize=40)
        ax.set_xticks(np.arange(len(x))+0.5)
        ax.set_yticks(np.arange(len(y))+0.5)
        #ax.set_yticklabels([''])
        ax.set_xlabel('Output Tokens', fontsize=60, fontweight='bold')
        ax.set_ylabel('Input Tokens', fontsize=60, fontweight='bold')
        #ax.set_title('Attention Heatmap', fontsize=40, fontweight='bold')

		# Save the plot in a pdf file
        plt.savefig('baseline/heatmap.pdf', format='pdf', dpi=300, bbox_inches='tight')
        
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
            for gen_id in generated_ids['sequences']
    ]
    return "".join(preds)

def get_rouge_and_bleu_scores (prediction, df_test_trimmed):
    rouge = evaluate.load('rouge')

#ROUGE_SCORE_RUNNING_AVG = np.zeros((3, 3), dtype=float) #i -> R1 R2 R3 j -> f p r
    #count = 0
    #score_log1 = tqdm(total=0, position=1, bar_format='{desc}')
    target = []
    for itr in tqdm(range (0, len(df_test_trimmed)), desc = 'Processing target'):
        target.append(df_test_trimmed['highlights'].iloc[itr])
    results = rouge.compute(predictions=prediction, references=target)
    print(results)
    """
    for itr in tqdm(range (0, len(df_test_trimmed)), desc = 'Processing Rouge scores'):
        stuff = df_test_trimmed['article'].iloc[itr]
        what_stuffs_supposed_to_be = df_test_trimmed['highlights'].iloc[itr]
        count+=1
        model_summary = summarizeText(trained_model, stuff)
        rouge_scores = rouge.get_scores(model_summary, what_stuffs_supposed_to_be)
        """"""
        splitted_highlights = what_stuffs_supposed_to_be.split()
        splitted_inference = model_summary.split()
        bleu_scores[0] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (1,0,0,0)) - bleu_scores[0])/count
        bleu_scores[1] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0,1,0,0)) - bleu_scores[1])/count
        bleu_scores[2] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0,0,1,0)) - bleu_scores[2])/count
        bleu_scores[3] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0,0,0,1)) - bleu_scores[3])/count
        bleu_scores[4] += (sentence_bleu(splitted_highlights, splitted_inference, weights = (0.25,0.25,0.25,0.25)) - bleu_scores[4])/count
       """"""
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
        
        score_log1.set_description_str("Rouge-1 Scores: f: {f1:4f}, p: {p1:4f}, r: {r1:4f} || Rouge-2 Scores: f: {f2:4f}, p: {p2:4f}, r: {r2:4f} || Rouge-L Scores: f: {f3:4f}, p: {p3:4f}, r: {r3:4f}".format(f1 = ROUGE_SCORE_RUNNING_AVG[0][0], p1 = ROUGE_SCORE_RUNNING_AVG[0][1], r1 = ROUGE_SCORE_RUNNING_AVG[0][2], f2 = ROUGE_SCORE_RUNNING_AVG[1][0], p2 = ROUGE_SCORE_RUNNING_AVG[1][1], r2 = ROUGE_SCORE_RUNNING_AVG[1][2], f3 = ROUGE_SCORE_RUNNING_AVG[2][0], p3 = ROUGE_SCORE_RUNNING_AVG[2][1], r3 = ROUGE_SCORE_RUNNING_AVG[2][2]))
        #score_log4.set_description_str("BLEU scores:: individual 1-gram : {b1:4f}, individual 2-gram : {b2:4f}, individual 3-gram : {b3:4f}, individual 4-gram : {b4:4f}, cumulative 4-gram : {b5:4f}".format(
#b1 = bleu_scores[0], b2 = bleu_scores[1], b3 = bleu_scores[2], b4 = bleu_scores[3], b5 = bleu_scores[4]))
"""
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
    

    #df_test_trimmed = pd.read_csv('CNN DailyMail Summarisation Data/test_stopwords.csv', encoding = "latin-1")
    #df_train_trimmed = pd.read_csv('CNN DailyMail Summarisation Data/train_stopwords.csv', encoding = "latin-1")
    #df_validation_trimmed = pd.read_csv('CNN DailyMail Summarisation Data/validation_stopwords.csv', encoding = "latin-1")
    
    #df_test_trimmed = pd.read_csv('CNN DailyMail Summarisation Data/test_preproc_no_stem.csv', encoding = "latin-1")
    #df_train_trimmed = pd.read_csv('CNN DailyMail Summarisation Data/train_preproc_no_stem.csv', encoding = "latin-1")
    #df_validation_trimmed = pd.read_csv('CNN DailyMail Summarisation Data/validation_preproc_no_stem.csv', encoding = "latin-1")
    
    #df_test_trimmed = pd.read_csv('CNN DailyMail Summarisation Data/test_preproc_stem.csv', encoding = "latin-1")
    df_test_trimmed = df_test_trimmed[:1]
    #df_train_trimmed = pd.read_csv('CNN DailyMail Summarisation Data/train_preproc_stem.csv', encoding = "latin-1")
    #df_validation_trimmed = pd.read_csv('CNN DailyMail Summarisation Data/validation_preproc_stem.csv', encoding = "latin-1")
    

    #data_module = NewsSummaryDataModule(df_train_trimmed, df_test_trimmed, df_validation_trimmed, tokenizer = tokenizer, batch_size = BATCH_SIZE)
    data_module = NewsSummaryDataModuleTest(df_test_trimmed, tokenizer = tokenizer, batch_size = BATCH_SIZE)
    
    trained_model = NewsSummaryModel.load_from_checkpoint(
        checkpoint_path="baseline/checkpoints/best-checkpoint.ckpt"
    )
    trained_model.freeze()
    

    sample_row = df_test_trimmed.sample(n=1).iloc[0]
    #text = sample_row['article']
    

    #text_input_ids = tokenizer.encode_plus(text, return_tensors='pt')['input_ids']
    
    #summary_input_ids = tokenizer.encode_plus(model_summary, return_tensors='pt')['input_ids']
    #print("size of summary_input_ids: ", sum
    logger = TensorBoardLogger("lightning_logs", name='news-summary')
    #trained_model.generate_attention_map(text_input_ids, summary_input_ids, text, model_summary)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=N_EPOCHS,
        accelerator = 'gpu',
        devices = 1
    )
    
    prediction = trainer.predict(model=trained_model, datamodule=data_module, return_predictions=True)
    #print("prediction: ", prediction)
    #get_rouge_and_bleu_scores(prediction, df_test_trimmed)
    df_train_trimmed['article'].iloc[0] = "Automatic text summarisation aims to produce a brief but comprehensive version of one or multiple documents, highlighting the most important information. There are two main summarisation techniques: extractive and abstractive. Extractive summarisation involves selecting key sentences from the original document, while abstractive summarisation involves creating new language based on the important information and requires a deeper understanding of the content."
    prediction = trainer.predict(model=trained_model, datamodule=data_module, return_predictions=True)
    text = "Automatic text summarisation aims to produce a brief but comprehensive version of one or multiple documents, highlighting the most important information. There are two main summarisation techniques: extractive and abstractive. Extractive summarisation involves selecting key sentences from the original document, while abstractive summarisation involves creating new language based on the important information and requires a deeper understanding of the content."

    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    
    outputs = trained_model.model.generate(input_ids=input_ids, max_length=100, num_beams=4, early_stopping=True)
    #print("output: ",outputs.keys())
    
    #print("output seq shape: ", outputs['sequences'].size())
    model_summary = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)

    print("Original Text: ", text)
    print("Generated Summary: ", model_summary)
    
if __name__ == "__main__":
        main()

