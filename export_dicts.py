import argparse
import json

import tensorflow as tf

def to_json(dictionary, path):
    s = json.dumps(dictionary)
    with open(path, 'w') as f:
        f.write(s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer', required=True, type=str)
    parser.add_argument('--wordindex', required=False, type=str, default=None)
    parser.add_argument('--indexword', required=False, type=str, default=None)  

    args = parser.parse_args()

    with open(args.tokenizer, 'r') as f:
        tok = tf.keras.preprocessing.text.tokenizer_from_json(f.read()) 

    if args.indexword is not None:
        to_json(tok.index_word, args.indexword) 
    if args.wordindex is not None:
        to_json(tok.word_index, args.wordindex)

