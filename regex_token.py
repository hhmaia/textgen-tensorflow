import argparse
import json

_regex_token = {
    r'\n' : ' \n ',
    r'\.' : ' . ',
    r'\,' : ' , ',
    r'\:' : ' : ',
    r'\!' : ' ! ',
    r'\?' : ' ? ',
    r'\;' : ' ; ',
    r'\"' : ' " ',
    r"\'" : " ' ",
    r'\(' : ' ( ',
    r'\)' : ' ) ',
    r'\{' : ' { ',
    r'\}' : ' } ',
    r'-' : ' -',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    
    args = parser.parse_args()
    with open(args.path, 'w') as f:
        f.write(json.dumps(_regex_token))
