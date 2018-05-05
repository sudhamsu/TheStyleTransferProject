import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMBED_TEXT = "../glove.6B.300d.txt"
EMBED_PATH = "../glove_300d.pickle"
EMBED_SIZE = 300


class LineReader:

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            yield line


def compute_cosine_similarity(source_vector, target_vector):
    """
    :param source_vector: source vector of shape (num_samples, num_features)
    :param target_vector: target vector of shape (num_samples, num_features)
    :return: cosine similarity between the two
    """
    return cosine_similarity(source_vector, target_vector)
    pass


def extract_embeddings():
    # glove_input_file = EMBED_TEXT
    # word2vec_output_file = EMBED_PATH
    # glove2word2vec(glove_input_file, word2vec_output_file)

    print("Extracting Glove Embeddings from text file and saving them as a dictionary")

    embeddings = {}
    reader = LineReader(EMBED_TEXT)
    for line in reader:
        words = line.split()
        if len(words) > 0:
            if words[0] == '.':
                print(line)
            vector = [float(num) for num in words[len(words)-EMBED_SIZE:]]
            embeddings[' '.join(words[0:len(words)-EMBED_SIZE])] = vector

    with open(EMBED_PATH, 'wb') as f:
        pickle.dump(embeddings, f)


def get_sentence_embedding(sentence, glove):

    # load the Stanford GloVe model
    # = KeyedVectors.load_word2vec_format(EMBED_PATH, binary=False)
    # calculate: (king - man) + woman = ?

    sent_embedding = np.zeros(shape=(len(sentence), EMBED_SIZE), dtype=np.float32)
    for i, word in enumerate(sentence):
        sent_embedding[i, :] = glove[word] if word in glove else glove['unk']

    min_vec = list(np.min(sent_embedding, axis=0))
    max_vec = (np.max(sent_embedding, axis=0))
    avg_vec = (np.mean(sent_embedding, axis=0))
    #print(min_vec)

    final_embeddings = np.append(min_vec, avg_vec)
    final_embeddings = np.append(final_embeddings, max_vec)
    final_embeddings = np.reshape(final_embeddings, (1, -1))

    return final_embeddings


def compute_content_preservation(data):
    """
    :param data: A list of tuples of source and target sentences
    :return: a list of cosine similarity between each pair
    """
    if not os.path.exists(EMBED_PATH):
        extract_embeddings()

    print("\nLoading word embeddings")
    with open(EMBED_PATH, 'rb') as f:
        glove = pickle.load(f)

    similarity = []
    for source, target in data:
        source_embedding = get_sentence_embedding(source, glove)
        target_embedding = get_sentence_embedding(target, glove)
        similarity.append(cosine_similarity(source_embedding, target_embedding)[0, 0])

    return similarity
