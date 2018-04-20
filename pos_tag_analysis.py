from nltk.tag import pos_tag
import matplotlib.pyplot as plt
from datautils import *
import os
from collections import defaultdict

genres = os.listdir(PATH)

tags_per_genre = defaultdict(lambda: defaultdict(int))        #{genre: {tag: count}}
all_tags = set()

for g in genres:
    print("genre: {}".format(g))
    gpath = os.path.join(PATH, g)
    for a in os.listdir(gpath):
        apath = os.path.join(gpath, a)
        tokenized_sentences = document_tokenize(apath)
        for sent in tokenized_sentences:
            #print(sent)
            tags = pos_tag(sent)
            for word, tag in tags:
                tags_per_genre[g][tag] += 1
                all_tags.update([tag])

TAGS = list(all_tags)
indices = [x for x in range(len(TAGS))]
shift = {0:-0.5,1:-0.25,2:0,3:0.25,4:0.5}
color={'Adventure':'b','Horror': 'k', 'Humor':'c', 'Fantasy':'r', 'Detective_Fiction':'g'}
for j, (key, values) in enumerate(tags_per_genre.items()):
    X = [i-shift[j] for i in indices]
    Y = [values[TAGS[x]] for x in indices]
    plt.plot(X, Y, label=key, color=color[key])
plt.legend()
plt.show()
