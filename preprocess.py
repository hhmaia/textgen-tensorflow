import os
import sys
import argparse

import numpy as np
import tensorflow as tf

from functools import partial
from tensorflow.keras.preprocessing.text import Tokenizer 

def export_tokenizer(tok_fp, texts_fp, max_vocab_size=None):
    tok = Tokenizer(max_vocab_size, filters='')
    tok.fit_on_texts(texts_fp.readlines())
    tok_fp.write(tok.to_json())
    return tok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utility for saving a tokenizer as json')

    parser.add_argument('-o', '--tok_path', type=partial(open, mode='w'),
        required=False,
        default='tokenizer.json',
        help='file name to save the json output')

    parser.add_argument('-p', '--texts_path', type=open, required=False, default=sys.stdin,
        help='input from which to generate the tokenizer') 

    parser.add_argument('-v', '--max_vocab_size', type=int, required=False, default=None,
        help='Maximum vocabulary size')

    args = parser.parse_args()
    export_tokenizer(args.tok_path, args.texts_path, args.max_vocab_size)
