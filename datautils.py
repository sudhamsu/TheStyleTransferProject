import re
import nltk

PATH = './Gutenberg'

def document_tokenize(path):
    with open(path, 'r') as f:
        text = f.read()
        text = re.sub('\n', ' ', text)
        text = re.sub('([()[\]{}])', r' \1 ', text)
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = []
    for s in sentences:
        tokenized_sentences.append(tokenize_sent(s))
    return tokenized_sentences


def tokenize_sent(s):
    tokens = re.split('(\w+)', s)
    tokens = [w.strip().lower() for w in tokens if w not in ['', ' ']]
    return tokens


def print_dict(d):
    print('-'*50)
    for key, val in d.items():
        print(key + '\t' + str(val))
