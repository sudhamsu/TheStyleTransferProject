import numpy
import gensim
from collections import defaultdict
from gensim.parsing.preprocessing import STOPWORDS
import datautils

# # dummy data
# sents = ["Human machine interface for lab abc computer applications human",
#          "A survey of user opinion of computer system response time",
#          "The EPS user interface management system",
#          "System and human system engineering testing of EPS",
#          "Relation of user perceived response time to error measurement",
#          "The generation of random binary unordered trees",
#          "The intersection graph of paths in trees",
#          "Graph minors IV Widths of trees and well quasi ordering",
#          "Graph minors A survey"]
# for i, s in enumerate(sents):
#     sents[i] = s.lower().split()
# # at this stage we have list of sentences, and each sentence is a list of words

doc_paths = ['Gutenberg/Adventure/Edgar_Rice_Burrough.txt',
             'Gutenberg/Adventure/Henry_Rider_Haggard.txt',
             'Gutenberg/Adventure/Jack_London.txt',
             'Gutenberg/Adventure/Robert_Louis_Stevenson.txt',
             'Gutenberg/Adventure/Rudyard_Kipling.txt',
             'Gutenberg/Detective_Fiction/Anna_Katharine_Green.txt',
             'Gutenberg/Detective_Fiction/Emile_Gaboriau.txt',
             'Gutenberg/Detective_Fiction/R_Austin_Freeman.txt',
             'Gutenberg/Detective_Fiction/Sir_Arthur_Conan_Doyle.txt',
             'Gutenberg/Detective_Fiction/Wilkie_Collins.txt',
             'Gutenberg/Fantasy/Edward_John_Moreton.txt',
             'Gutenberg/Fantasy/Howard_Pyle.txt',
             'Gutenberg/Fantasy/James_Branch_Cabell.txt',
             'Gutenberg/Fantasy/Lyman_Frank_Baum.txt',
             'Gutenberg/Fantasy/William_Morris.txt',
             'Gutenberg/Horror/Algernon_Blackwood.txt',
             'Gutenberg/Horror/Ambrose_Bierce.txt',
             'Gutenberg/Horror/Bram_Stoker.txt',
             'Gutenberg/Horror/Edgar_Allan_Poe.txt',
             'Gutenberg/Horror/Henry_James.txt',
             'Gutenberg/Humor/Finley_Peter_Dunne.txt',
             'Gutenberg/Humor/Jerome_Klapka_Jerome.txt',
             'Gutenberg/Humor/John_Kendrick_Bangs.txt',
             'Gutenberg/Humor/P_G_Wodehouse.txt',
             'Gutenberg/Humor/Stephen_Leacock.txt']


class Corpus(object):
    def __init__(self, list_of_docs, clip_docs=None):
        self.list_of_docs = list_of_docs
        self.dictionary = self.generate_dictionary()
        self.clip_docs = clip_docs

    def generate_dictionary(self):
        doc_stream = (tokens for tokens in self.iter_docs())
        dictionary = gensim.corpora.Dictionary(doc_stream)

        # ignore words that appear in less than 20 documents or more than 10% documents
        # dictionary.filter_extremes(no_below=20, no_above=0.1)
        dictionary.filter_extremes(no_below=4, no_above=0.5)

        return dictionary

    def iter_docs(self):
        for doc in self.list_of_docs:
            doc_sentences = datautils.document_tokenize(doc)
            tokens = [item for sublist in doc_sentences for item in sublist]
            yield tokens
        # for sent in sents:
        #     yield sent

    def __iter__(self):
        for doc in self.list_of_docs:
            doc_sentences = datautils.document_tokenize(doc)
            tokens = [item for sublist in doc_sentences for item in sublist]
            tokens = [word for word in tokens if word not in STOPWORDS]
            yield self.dictionary.doc2bow(tokens)
        # for sent in sents:
        #     tokens = [word for word in sent if word not in STOPWORDS]
        #     yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs


corpus = Corpus(doc_paths)
print('Dictionary size:', len(corpus.dictionary))
# for vec in iter(corpus):
#     for i, j in vec:
#         print('('+corpus.dictionary[i]+', '+str(i)+', '+str(j)+')', end=' ')
#     print()

lda_model = gensim.models.LdaModel(corpus, num_topics=10, id2word=corpus.dictionary, passes=5)
top_words = [[word for word, _ in lda_model.show_topic(topicno, topn=10)] for topicno in range(lda_model.num_topics)]
print(top_words)
