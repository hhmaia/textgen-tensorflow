import os
import sys
import argparse

import numpy as np
import tensorflow as tf

from functools import partial
from mlutils import export_vocabulary
from tensorflow.keras.preprocessing.text import Tokenizer 


def export_tokenizer(tok_fp, texts_fp, max_vocab_size=None):
    tok = Tokenizer(max_vocab_size, filters='')
    tok.fit_on_texts(texts_fp.readlines())
    tok_fp.write(tok.to_json())
    return tok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utility for creating and saving a tokenizer as json.')
    parser.add_argument('-o', '--tok_path',
            type=partial(open, mode='w'), required=False, default=sys.stdout,
            help='File name to save the json output. Defaults to stdout.')
    parser.add_argument('-p', '--texts_path',
            type=open, required=False, default=sys.stdin,
            help='File to generate the tokenizer from. Defaults to stdin.') 
    parser.add_argument('-n', '--max_vocab_size',
            type=int, required=False, default=None,
            help='Maximum vocabulary size. Default is full vocabulary.')
    parser.add_argument('-v', '--vocab_path',
            type=str, required=False, default=None,
            help='If specified, saves a file with the vocabulary, one word per line') 

    args = parser.parse_args()
    tok = export_tokenizer(args.tok_path, args.texts_path, args.max_vocab_size)
    if args.vocab_path is not None:
        export_vocabulary(args.vocab_path, args.max_vocab_size, tok.word_index)

