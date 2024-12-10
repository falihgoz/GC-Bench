# references:
# [1] https://github.com/DART-Laboratory/Flash-IDS

import gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from multiprocessing import Pool
# from itertools import compress
# from tqdm import tqdm
# import time
import math
import torch
import numpy as np


# ref. [1], [2]
class EpochSaver(CallbackAny2Vec):
    def __init__(self, target_save_file: str):
        self.epoch = 0
        self.target_save_file = target_save_file

    def on_epoch_end(self, model):
        model.save(self.target_save_file)
        self.epoch += 1

# ref. [1]
class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

# ref. [1]
class PositionalEncoder:
    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]

def train_w2v_model(save_model_file: str, sentences):
    # w2v_model = Word2Vec(sentences=phrases, vector_size=30, window=5, min_count=1, workers=8,epochs=300,callbacks=[saver,logger])
    
    logger = EpochLogger()
    saver = EpochSaver(save_model_file)
    
    w2v_model = Word2Vec(sentences=sentences, vector_size=30, window=5, min_count=1, workers=8,epochs=300,callbacks=[saver,logger])
    
    return w2v_model

def load_w2v_model(save_model_file: str):
    return Word2Vec.load(save_model_file)

# ref. [1]
def infer(document, w2v_model, encoder_model):
    word_embeddings = [w2v_model.wv[word] for word in document if word in  w2v_model.wv]
    
    if not word_embeddings:
        return np.zeros(20)

    output_embedding = torch.tensor(word_embeddings, dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder_model.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)




