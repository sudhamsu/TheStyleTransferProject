import numpy as np
import datautils as du
import pickle
import os


MAX_LENGTH = 20
TRAIN_SENTS_PER_STYLE = 5000
TEST_SENTS_PER_STYLE = 1000

filename_prepend = 'horror_humor_'
styles = ["../Gutenberg/Horror/horror.txt", "../Gutenberg/Humor/humor.txt"]
sents_per_style = TRAIN_SENTS_PER_STYLE + TEST_SENTS_PER_STYLE
data = [(a, sent)
        for a, style in enumerate(styles)
        for sent in np.random.choice(du.document_tokenize(style, max_length=MAX_LENGTH, tokenize_words=True),
                                     sents_per_style, replace=False)]

train_data = []
test_data = []
for a in range(len(styles)):
    start = a * sents_per_style
    test_start = a * sents_per_style + TRAIN_SENTS_PER_STYLE
    end = (a + 1) * sents_per_style
    train_data += data[start:test_start]
    test_data += data[test_start:end]

save_dir = 'data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pickle.dump(train_data, open(save_dir+'/'+filename_prepend+'train.pkl', 'wb'))
pickle.dump(test_data, open(save_dir+'/'+filename_prepend+'test.pkl', 'wb'))



# build human evaluation test data
np.random.shuffle(test_data)
final_test_data = []

# test manually if you can guess the style
# keep the line in the set if you can guess correctly
# else delete it
counters = [0, 0]
for a, line in test_data:
    print('\n'+' '.join(line))
    guess = input('Guess: ')
    if guess == str(a):
        print('Correct! Saving line to test set.')
        final_test_data.append((a, line))
        counters[a] += 1
        print('Test set counts:', counters)
    elif guess == 'skip':
        print('Skipping... It was ', a)
    elif guess == 'stop':
        print('Stopping, and saving test set built so far.')
        break
    else:
        print('Incorrect! Moving on...')

pickle.dump(final_test_data, open(save_dir+'/'+filename_prepend+'test_human.pkl', 'wb'))
