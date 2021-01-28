import nltk
import random
import pickle
import sys
import numpy as np



TRAIN_PROCESSED_FILE = '../clean_data/training-processed-lemmatized.csv'
TEST_PROCESSED_FILE = '../clean_data/test-processed-lemmatized.csv'
USE_BIGRAMS = True
USE_TRIGRAMS = True

numIterations = 1
Train = True


def save_results_to_csv(results, csv_file):

    with open(csv_file, 'w') as csv:
        csv.write('id,prediction\n')
        for tweet_id, pred in results:
            csv.write(tweet_id)
            csv.write(',')
            csv.write(str(pred))
            csv.write('\n')

def get_data_from_file(file_name, isTrain=True):
    data = []
    with open(file_name, 'r') as csv:
        lines = csv.readlines()
        #total = len(lines)
        for  line in lines:
            if isTrain:
                
                tag = line.split(',')[1]
                bag_of_words = line.split(',')[2].split()
                if USE_BIGRAMS:
                    bag_of_words_bigram = list(nltk.bigrams(line.split(',')[2].split()))
                    bag_of_words = bag_of_words+bag_of_words_bigram
                if USE_TRIGRAMS:
                    bag_of_words_trigram = list(nltk.trigrams(line.split(',')[2].split()))
                    bag_of_words = bag_of_words+bag_of_words_trigram
            else :
                tag = '5'
                bag_of_words = line.split(',')[1].split()
                if USE_BIGRAMS:
                    bag_of_words_bigram = list(nltk.bigrams(line.split(',')[1].split()))
                    bag_of_words = bag_of_words+bag_of_words_bigram
                if USE_TRIGRAMS:
                    bag_of_words_trigram = list(nltk.trigrams(line.split(',')[1].split()))
                    bag_of_words = bag_of_words+bag_of_words_trigram
            data.append((bag_of_words, tag))
           
    return data

def split_data(tweets, validation_split=0.1):
    index = int((1 - validation_split) * len(tweets))
    random.shuffle(tweets)
    return tweets[:index], tweets[index:]

def list_to_dict(words_list):
    return dict([(word, True) for word in words_list])

if __name__ == '__main__':
    train = True
    np.random.seed(1337)
    train_csv_file = TRAIN_PROCESSED_FILE
    test_csv_file = TEST_PROCESSED_FILE
    train_data = get_data_from_file(train_csv_file, isTrain=True)
    train_set, validation_set = split_data(train_data)
    training_set_formatted = [(list_to_dict(element[0]), element[1]) for element in train_set]
    validation_set_formatted = [(list_to_dict(element[0]), element[1]) for element in validation_set]
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[1]
    classifier = nltk.MaxentClassifier.train(training_set_formatted, algorithm, max_iter=numIterations)
    classifier.show_most_informative_features(10)
    count = int(0)
    for review in validation_set_formatted:
        label = review[1]
        text = review[0]
        determined_label = classifier.classify(text)
        #print(determined_label, label)
        if determined_label!=label:
            count+=int(1)
    print (count)
    print (len(validation_set))
    accuracy = (len(validation_set)-count)/len(validation_set)
    print ('Validation set accuracy:%.4f'%accuracy )
    f = open('maxEnt_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
    print ('\nPredicting for test data')
    test_data = get_data_from_file(test_csv_file, isTrain=False)
    test_set_formatted = [(list_to_dict(element[0]), element[1]) for element in test_data]
    tweet_id = int(0)
    results = []
    for review in test_set_formatted:
        text = review[0]
        label = classifier.classify(text)
        results.append((str(tweet_id), label))
        tweet_id += int(1)
    save_results_to_csv(results, 'maxent.csv')
    print ('\nSaved to maxent.csv')