from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
from datasets import load_dataset

import torch
import random
import warnings
from torch.utils.data.distributed import DistributedSampler

import pandas as pd

warnings.filterwarnings('ignore')

class DatasetLoader(Dataset):
    def __init__(self, args, tokenizer, src, method=None):
        super().__init__()
        
        self.args = args
        
        text = '' 
    
        if src == 'validation':
            methods = ['wikitext', 'ptb']
            methods = ['wiki'] 
            for method in methods:
                # if method == 'wikitext':
                #     # Load the WikiText-2 validation dataset
                #     dataset = load_dataset('wikitext', 'wikitext-2-v1')
                #     validation_data = dataset['validation']
                #     # Extract text from the dataset
                #     t = [entry['text'] for entry in validation_data]
                # elif method == 'ptb':
                #     # Load the Penn Treebank validation dataset
                #     dataset = load_dataset('ptb_text_only')
                #     validation_data = dataset['validation']
                #     # Extract text from the dataset
                #     t = [entry['sentence'] for entry in validation_data]
                # elif method == 'c4':
                #     # Load the C4 validation dataset
                #     dataset = load_dataset('c4', 'en')
                #     validation_data = dataset['validation']
                #     # Extract text from the dataset
                #     t = [entry['text'] for entry in validation_data]
                file = os.path.join('dataset', f'{method}.txt')
                
                with open(file, 'r') as f:
                    t = f.readlines()
                
                text += "\n\n".join(t)
                text += "\n\n\n" 
        else:
            if method is None:
                method = args.method
                
            file = os.path.join('dataset', f'{method}.txt')
                
            with open(file, 'r') as f:
                t = f.readlines()
            
            text += "\n\n".join(t)
        
        self.corpus = tokenizer(text, return_tensors='pt').input_ids
        self.corpus = torch.concat([self.corpus, self.corpus[:, -self.args.max_seq_len+self.corpus.numel() % self.args.max_seq_len:]], dim=1)
        self.corpus = self.corpus.reshape([-1, self.args.max_seq_len])
        
    def __len__(self):
        #return self.corpus.numel() // self.args.max_seq_len
        return len(self.corpus)

    def __getitem__(self, idx):
        #return self.corpus[0, idx*self.args.max_seq_len:((idx+1)*self.args.max_seq_len)]
        return self.corpus[idx]
    

class Data:
    def __init__(self, args, tokenizer):
        
        self.train_dataset = DatasetLoader(args, tokenizer, 'train')
   
        self.loader_train = DataLoader(
                    self.train_dataset, 
                    batch_size=args.train_batch_size, shuffle=True, 
                    num_workers=2
                    )
        
        val_dataset = DatasetLoader(args, tokenizer, 'validation')
   
        self.loader_validation = DataLoader(
                    val_dataset, 
                    batch_size=args.eval_batch_size, shuffle=False, 
                    num_workers=2
                    )
        
        
        self.loader_test = dict()
        
        for method in ['wikitext', 'ptb']: #, 'c4'
            test_dataset = DatasetLoader(args, tokenizer, 'test', method)
       
            self.loader_test[method] = DataLoader(
                        test_dataset, 
                        batch_size=args.eval_batch_size, shuffle=False, 
                        num_workers=2
                        )