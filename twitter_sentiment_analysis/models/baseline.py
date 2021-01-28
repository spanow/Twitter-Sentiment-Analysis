

# Classifies a tweet based on the number of positive and negative words in it

STEMMER = False
LEMMATIZER = True
TRAIN = True


if STEMMER and LEMMATIZER:
    TRAIN_PROCESSED_FILE = '../clean_data/training-processed-stemmed-lemmatized.csv'
    TEST_PROCESSED_FILE = '../clean_data/test-processed-stemmed-lemmatized.csv'
    POSITIVE_WORDS_FILE = '../dataset/positive-words.txt'
    NEGATIVE_WORDS_FILE = '../dataset/negative-words.txt'
elif LEMMATIZER:
    TRAIN_PROCESSED_FILE = '../clean_data/training-processed-lemmatized.csv'
    TEST_PROCESSED_FILE = '../clean_data/test-processed-lemmatized.csv'
    POSITIVE_WORDS_FILE = '../dataset/positive-words.txt'
    NEGATIVE_WORDS_FILE = '../dataset/negative-words.txt'
elif STEMMER:
    TRAIN_PROCESSED_FILE = '../clean_data/training-processed-stemmed.csv'
    TEST_PROCESSED_FILE = '../clean_data/test-processed-stemmed.csv'
    POSITIVE_WORDS_FILE = '../dataset/positive-words.txt'
    NEGATIVE_WORDS_FILE = '../dataset/negative-words.txt'
else :
    TRAIN_PROCESSED_FILE = '../clean_data/training-processed.csv'
    TEST_PROCESSED_FILE = '../clean_data/test-processed.csv'
    POSITIVE_WORDS_FILE = '../dataset/positive-words.txt'
    NEGATIVE_WORDS_FILE = '../dataset/negative-words.txt'




def classify(processed_csv, test_file=True, **params):
    positive_words = file_to_wordset(params.pop('positive_words'))
    negative_words = file_to_wordset(params.pop('negative_words'))
    predictions = []
    with open(processed_csv, 'r') as csv:
        for line in csv:
            if test_file:
                tweet_id, tweet = line.strip().split(',')
            else:
                tweet_id, label, tweet = line.strip().split(',')
            pos_count, neg_count = 0, 0
            for word in tweet.split():
                if word in positive_words:
                    pos_count += 1
                elif word in negative_words:
                    neg_count += 1
            # print pos_count, neg_count
            prediction = 1 if pos_count >= neg_count else 0
            if test_file:
                predictions.append((tweet_id, prediction))
            else:
                predictions.append((tweet_id, int(label), prediction))
    return predictions


def file_to_wordset(filename):
    words = []
    with open(filename, 'r') as f:
        for line in f:
            words.append(line.strip())
    return set(words)

def save_results_to_csv(results, csv_file):
    with open(csv_file, 'w') as csv:
        csv.write('id,prediction\n')
        for tweet_id, pred in results:
            csv.write(tweet_id)
            csv.write(',')
            csv.write(str(pred))
            csv.write('\n')

def save_results_to_csv(results, csv_file):  
    with open(csv_file, 'w') as csv:
        csv.write('id,prediction\n')
        for tweet_id, pred in results:
            csv.write(tweet_id)
            csv.write(',')
            csv.write(str(pred))
            csv.write('\n')



if __name__ == '__main__':
    if TRAIN:
        predictions = classify(TRAIN_PROCESSED_FILE, test_file=(not TRAIN), positive_words=POSITIVE_WORDS_FILE, negative_words=NEGATIVE_WORDS_FILE)
        correct = sum([1 for p in predictions if p[1] == p[2]]) * 100.0 / len(predictions)
        print 'Correct = %.2f%%' % correct
    else:
        predictions = classify(TEST_PROCESSED_FILE, test_file=(not TRAIN), positive_words=POSITIVE_WORDS_FILE, negative_words=NEGATIVE_WORDS_FILE)
        save_results_to_csv(predictions, 'baseline.csv')
