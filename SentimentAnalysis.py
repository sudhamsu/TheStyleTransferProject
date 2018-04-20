import numpy as np
import datautils
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

genres = ['Adventure',
          'Detective',
          'Fantasy',
          'Horror',
          'Humor']
authors = [['Edgar Rice Burrough',
            'Henry Rider Haggard',
            'Jack London',
            'Robert Louis Stevenson',
            'Rudyard Kipling'],
           ['Anna Katharine Green',
            'Emile Gaboriau',
            'R Austin Freeman',
            'Sir Arthur Conan Doyle',
            'Wilkie Collins'],
           ['Edward John Moreton',
            'Howard Pyle',
            'James Branch Cabell',
            'Lyman Frank Baum',
            'William Morris'],
           ['Algernon Blackwood',
            'Ambrose Bierce',
            'Bram Stoker',
            'Edgar Allan Poe',
            'Henry James'],
           ['Finley Peter Dunne',
            'Jerome Klapka Jerome',
            'John Kendrick Bangs',
            'P G Wodehouse',
            'Stephen Leacock']]
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

doc_sentences = []
doc_lengths = []
doc_sentence_scores = []
for i in range(5):
    sents = []
    sent_lengths = []
    for j in range(5*i, 5*(i+1)):
        author_sents = datautils.document_tokenize(doc_paths[j], tokenize_words=False)
        sents.extend(author_sents)
        sent_lengths.append(len(author_sents))
    doc_sentences.append(sents)
    doc_lengths.append(sent_lengths)
    doc_sentence_scores.append(np.zeros([len(sents), 4]))

sid = SentimentIntensityAnalyzer()
for i in range(5):
    for j, s in enumerate(doc_sentences[i]):
        scores = sid.polarity_scores(s)
        doc_sentence_scores[i][j, :] = np.array([scores['compound'], scores['neg'], scores['neu'], scores['pos']])

np.save('sentiment_scores', doc_sentence_scores)
np.save('doc_lengths', doc_lengths)

plt.figure()
plt.hist([doc_sentence_scores[i][:, 0] for i in range(5)], np.linspace(-1, 1, 9), label=genres, density=True)
plt.xlabel('VADER Compound Sentiment Score')
plt.ylabel('Probability Density')
plt.title('Sentiment Analysis by Genre')
plt.legend(loc=2)
plt.tight_layout()
plt.savefig('plots/sentiments.png')
plt.close()

for g in range(len(genres)):
    plt.figure()
    plt.hist([doc_sentence_scores[g][np.sum(doc_lengths[g][0:i]):np.sum(doc_lengths[g][0:i+1]), 0] for i in range(5)], np.linspace(-1, 1, 9), label=authors[g], density=True)
    plt.xlabel('VADER Compound Sentiment Score')
    plt.ylabel('Probability Density')
    plt.title('Sentiment Analysis by Author in '+genres[g]+' Genre')
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig('plots/sentiments_'+genres[g].lower()+'.png')
    plt.close()
