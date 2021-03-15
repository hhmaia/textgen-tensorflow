import os 
import re
import codecs
import json
import argparse

from regex_token import _regex_token


def replace_tokens(s, regex_token=_regex_token): 
    x = s.lower()
    for p, t in regex_token:
        x = re.sub(p, t, x) 
    return x


def verses_to_tape(verses):
    return ' \n '.join(replace_tokens(verse) for verse in verses) 


def json_to_verses(path):
    books = json.load(codecs.open(path, 'r', 'utf-8-sig')) 
    return [verse for book in books 
        for chapter in book['chapters']
        for verse in chapter]


def json_to_tape(path):
    return verses_to_tape(json_to_verses(path))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser('Preprocessing tool for bible on json format.')
    parser.add_argument('-p', '--path', type=str, required=True,
            help='Path for the json file.')
    parser.add_argument('-v', '--verses', action='store_true',
            help='Output is one line per verse instead of tape.')
    args = parser.parse_args()
    try:
        if args.verses:
            for line in json_to_verses(args.path):
                print(line)
        else: 
            print(json_to_tape(args.path))

    except BrokenPipeError:
        pass
        # its ok on this case

