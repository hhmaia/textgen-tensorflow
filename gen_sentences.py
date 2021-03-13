import sys
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from parsecorpus import replace_tokens  


def gen(model_path, seeds_fp, tokenizer_fp, slen, n):
    seeds = [replace_tokens(seed.strip('\n'))
             for seed in seeds_fp.readlines()]
    tokenizer = tokenizer_from_json(tokenizer_fp.read())
    seed_seqs = tokenizer.texts_to_sequences(seeds)
#    padded_seqs = pad_sequences(seed_seqs, slen)
    
    model : Model = load_model(model_path)
    model.summary()

    feedback = list(np.zeros((32,)))
    for seq in seed_seqs:
        feedback.extend(list(seq))
        for _ in range(1):
            logit = 0
            while logit not in [tokenizer.word_index['.']]:
                seq_input = np.expand_dims(feedback[-slen:], 0)
                pred = model.predict([seq_input], 1)
                logit = pred.squeeze().argmax()
                feedback.append(logit)
        feedback.append(tokenizer.word_index['\n'])

    out_seq = tokenizer.sequences_to_texts([feedback])
    print(out_seq[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentences generation script.')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-s', '--seed', type=open, default=sys.stdin, required=False)
    parser.add_argument('-t', '--tokenizer', type=open, required=True) 
    parser.add_argument('-l', '--seq_len', type=int, required=True)
    parser.add_argument('-n', '--num_words', type=int, default='1', required=False)
    args = parser.parse_args()
    gen(args.model, args.seed, args.tokenizer, args.seq_len, args.num_words)
