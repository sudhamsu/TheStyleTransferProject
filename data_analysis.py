from datautils import *
import os
from collections import defaultdict

genres = os.listdir(PATH)

vocab_per_author = defaultdict(lambda: defaultdict(int))    #{author: {word:count}}
vocab_per_genre = defaultdict(lambda: defaultdict(int))     #{genre: {word:count}}
sentences_per_genre = defaultdict(int)
sentences_per_author = defaultdict(int)
tokens_per_genre = defaultdict(int)
tokens_per_author = defaultdict(int)

for g in genres:
    print("Genre: {}".format(g))
    gpath = os.path.join(PATH, g)
    authors = os.listdir(gpath)
    for a in authors:
        authorpath = os.path.join(gpath, a)
        tokenized_data = document_tokenize(authorpath)
        for sent in tokenized_data:
            sentences_per_author[g+'_'+a] += 1
            sentences_per_genre[g] += 1
            for word in sent:
                vocab_per_author[g+'_'+a][word] += 1
                vocab_per_genre[g][word] += 1
                tokens_per_author[g+'_'+a] += 1
                tokens_per_genre[g] += 1
    #print(vocab_per_genre[g])
    #print('Number of words in the genre {} are: {}'.format(g, len(vocab_per_genre[g])))
print("Number of sentences in each genre")
print_dict(sentences_per_genre)
print("\nNumber of sentences for each author")
print_dict(sentences_per_author)
print("\nNumber of unique words in each genre")
print_dict({genre: len(val) for genre, val in vocab_per_genre.items()})
print("\nNumber of unique words for each author")
print_dict({genre: len(val) for genre, val in vocab_per_author.items()})
print("\nNumber of tokens per genre")
print_dict(tokens_per_genre)
print("\nNumber of tokens for each author")
print_dict(tokens_per_author)
