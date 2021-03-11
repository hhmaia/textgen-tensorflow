import os 
import re
import codecs
import json
import argparse

from regex_token import _regex_token


def _replace_tokens(s, regex_token=_regex_token): 
    x = s.lower()
    for p, t in regex_token:
        x = re.sub(p, t, x) 
    return x


def _json_verses(path):
    return [verse for book in json.load(codecs.open(path, 'r', 'utf-8-sig')) 
        for chapter in book['chapters']
        for verse in chapter]

def _verses_tape(verses):
    return ''.join(_replace_tokens(verse) for verse in verses) 


def _json_tape(path):
    return _verses_tape(_json_verses(path))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser('preprocessing tool for bible on json format')
    parser.add_argument('-p', '--path', type=str, required=True, help='path for the json file')
    parser.add_argument('-v', '--verses', action='store_true',
            help='output is one line per verse instead of tape')
    args = parser.parse_args()
    try:
        if args.verses:
            for line in _json_verses(args.path):
                print(line)
        else: 
            print(_json_tape(args.path))

    except BrokenPipeError:
        pass
        # its ok on this case

