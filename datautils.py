import re
import nltk

PATH = './Gutenberg'


def document_tokenize(path, max_length=None, tokenize_words=True):
    with open(path, 'r') as f:
        text = f.read()
        text = re.sub('\n', ' ', text)
        text = re.sub('([()[\]{}",_;])', r' \1 ', text)
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = []
    if tokenize_words:
        for s in sentences:
            tokenized_sent = tokenize_sent(s)
            if max_length is None:
                tokenized_sentences.append(tokenized_sent)
            else:
                if len(tokenized_sent) < max_length:
                    tokenized_sentences.append(tokenized_sent)
        return tokenized_sentences
    else:
        return sentences


def tokenize_sent(s):
    tokens = re.split('(\w+)', s)
    tokens = [w.strip().lower() for w in tokens if w.strip() != '']
    tokens = [word for t in tokens for word in t.split()]
    return tokens


def print_dict(d):
    print('-'*50)
    for key, val in d.items():
        print(key + '\t' + str(val))
