import os
import re
import numpy as np
from datautils import *
import pickle
import nltk
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import defaultdict

genres = sorted(os.listdir('./Gutenberg'))
authors = [sorted(os.listdir(os.path.join('./Gutenberg', g))) for g in genres]
tagdict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
tags = sorted(tagdict.keys())
tags += ['#']


def save_tags():
    tags_per_genre = {}          #for each genre, list of part of speech tag counts
    tags_per_author = {}        #for each genre, a dictionary of each author and their tag distribution
    for i, g in enumerate(genres):
        gtags = [0]*len(tags)
        gatags = {}
        for a in authors[i]:
            print(g, a)
            aname = re.findall('(.*)\.txt', a)[0]
            atags = [0]*len(tags)
            path = './Gutenberg/{}/{}'.format(g, a)
            sentences = document_tokenize(path)
            for s in sentences:
                tagset = nltk.tag.pos_tag(s)
                for word, tag in tagset:
                    try:
                        #print(tag)
                        gtags[tags.index(tag)] += 1
                        atags[tags.index(tag)] += 1
                    except:
                        print("Tag not found: {}".format(tag))
            gatags[aname] = atags
        tags_per_author[g] = gatags
        tags_per_genre[g] = gtags
    with open('tags_per_genre.pkl', 'wb') as f:
        pickle.dump(tags_per_genre, f)
    with open('tags_per_author.pkl', 'wb') as f:
        pickle.dump(tags_per_author, f)


def load_tags():
    try:
        with open('tags_per_genre.pkl', 'rb') as f:
            tags_per_genre = pickle.load(f)
    except:
        tags_per_genre = None

    with open('tags_per_author.pkl', 'rb') as f:
        tags_per_author = pickle.load(f)

    return tags_per_genre, tags_per_author


def plot_per_genre(tags_per_genre):
    colors = ['b','g','m','c','k']
    total = np.zeros(len(tags))
    plt.figure()
    for i, g in enumerate(genres):
        tag_distribution = np.array(tags_per_genre[g])
        total_count = sum(tag_distribution)
        tag_probability = np.divide(tag_distribution, total_count)
        total += tag_probability
        plt.plot(np.arange(len(tags)), tag_probability, c=colors[i], label=g)
    plt.legend()
    plt.xlabel('POS Tags')
    plt.ylabel('Probability of Tag')
    plt.xticks(np.arange(len(tags)), tags, rotation='vertical')
    plt.tight_layout()
    plt.savefig('tags_per_genre.png')

    plt.figure()
    avg = total / len(genres)
    for i, g in enumerate(genres):
        tag_distribution = np.array(tags_per_genre[g])
        total_count = sum(tag_distribution)
        tag_probability = np.divide(tag_distribution, total_count)
        plt.plot(np.arange(len(tags)), tag_probability-avg, c=colors[i], label=g)
    plt.legend()
    plt.xlabel('POS Tags')
    plt.ylabel('Deviation from mean of probability of Tag')
    plt.xticks(np.arange(len(tags)), tags, rotation='vertical')
    plt.tight_layout()
    plt.savefig('tags_per_genre_mean.png')



def plot_per_author(tags_per_author):
    for g in genres:
        print(g)
        colors = ['b','g','m','c','k']
        total = np.zeros(len(tags))
        plt.figure()
        for i, a in enumerate(tags_per_author[g].keys()):
            tag_distribution = np.array(tags_per_author[g][a])
            total_count = sum(tag_distribution)
            tag_probability = np.divide(tag_distribution, total_count)
            total += tag_probability
            plt.plot(np.arange(len(tags)), tag_probability, c=colors[i], label=a)
        plt.legend()
        plt.xlabel('POS Tags')
        plt.ylabel('Probability of Tag')
        plt.xticks(np.arange(len(tags)), [t.lower() for t in tags], rotation='vertical')
        plt.tight_layout()
        plt.savefig('tags_per_author_{}.png'.format(g))

        plt.figure()
        avg = total / len(tags_per_author[g].keys())
        for i, a in enumerate(tags_per_author[g].keys()):
            tag_distribution = np.array(tags_per_author[g][a])
            total_count = sum(tag_distribution)
            tag_probability = np.divide(tag_distribution, total_count)
            plt.plot(np.arange(len(tags)), tag_probability-avg, c=colors[i], label=a)
        plt.legend()
        plt.xlabel('POS Tags')
        plt.ylabel('Deviation from mean of probability of Tag')
        plt.xticks(np.arange(len(tags)), [t.lower() for t in tags], rotation='vertical')
        plt.tight_layout()
        plt.savefig('tags_per_author_{}_mean.png'.format(g))


if __name__ == '__main__':
    #save_tags()
    tags_per_genre, tags_per_author = load_tags()
    #print(tags_per_author)
    #plot_per_genre(tags_per_genre)
    #print(tags)
    plot_per_author(tags_per_author)
