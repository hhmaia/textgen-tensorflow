import sys
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from parsecorpus import replace_tokens  


def gen(model_path, seed_fp, tokenizer_fp, slen, n):
    tokenizer = tokenizer_from_json(tokenizer_fp.read())
    seed = seed_fp.read()
    seed_seqs = tokenizer.texts_to_sequences([replace_tokens(seed)])
    seed_seq = list(pad_sequences(seed_seqs, slen)[0])

    model : Model = load_model(model_path)
    model.summary()
    
    for _ in range(n):
        seq_input = np.expand_dims(seed_seq[-slen:], 0)
        pred = model.predict([seq_input], 1)
        logit = pred.squeeze().argmax()
        seed_seq.append(logit)

    out_sentences = tokenizer.sequences_to_texts([seed_seq])
    out_text = ''.join(out_sentences)
    print(out_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentences generation script.')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-s', '--seed', type=open, default=sys.stdin, required=False)
    parser.add_argument('-t', '--tokenizer', type=open, required=True) 
    parser.add_argument('-l', '--seq_len', type=int, required=True)
    parser.add_argument('-n', '--num_words', type=int, default='1', required=False)
    args = parser.parse_args()
    gen(args.model, args.seed, args.tokenizer, args.seq_len, args.num_words)
