from nltk import FreqDist
import pickle
import sys
import os
from collections import Counter


# Takes in a preprocessed CSV file and gives statistics
# Writes the frequency distribution of words and bigrams
# to pickle files.


train_data_file="training-processed-lemmatized.csv"

train_data_path = '../clean_data/'+train_data_file

def analyze_tweet(tweet):
    result = {}
    result['MENTIONS'] = tweet.count('USER_MENTION')
    result['URLS'] = tweet.count('URL')
    result['POS_EMOS'] = tweet.count('EMO_POS')
    result['NEG_EMOS'] = tweet.count('EMO_NEG')
    tweet = tweet.replace('USER_MENTION', '').replace('URL', '')
    words = tweet.split()
    
    result['WORDS'] = len(words)
    bigrams = get_bigrams(words)
    trigrams = get_Trigrams(words)
    result['BIGRAMS'] = len(bigrams)
    result['TRIGRAMS'] = len(trigrams)
    return result, words, bigrams , trigrams


def get_bigrams(tweet_words):
    bigrams = []
    num_words = len(tweet_words)
    for i in xrange(num_words - 1):
        bigrams.append((tweet_words[i], tweet_words[i + 1]))
    return bigrams

def get_Trigrams(tweet_words):
    trigrams = []
    num_words = len(tweet_words)
    for i in xrange(num_words - 2):
        trigrams.append((tweet_words[i], tweet_words[i + 1],tweet_words[i + 2]))
    return trigrams

def get_freqdist(bigrams):
    freq_dict = {}
    for bigram in bigrams:
        if freq_dict.get(bigram):
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1
    counter = Counter(freq_dict)
    return counter

    
def write_status(i, total):
    sys.stdout.write('\r')
    sys.stdout.write('Processing %d/%d' % (i, total))
    sys.stdout.flush()

if __name__ == '__main__':
    num_tweets, num_pos_tweets, num_neg_tweets = 0, 0, 0
    num_mentions, max_mentions = 0, 0
    num_emojis, num_pos_emojis, num_neg_emojis, max_emojis = 0, 0, 0, 0
    num_urls, max_urls = 0, 0
    num_words, num_unique_words, min_words, max_words = 0, 0, 1e6, 0
    num_bigrams,num_Trigrams, num_unique_bigrams = 0, 0 ,0
    all_words = []
    all_words_Tri = []
    all_bigrams = []
    all_Trigrams = []
    with open(train_data_path, 'r') as csv:
        lines = csv.readlines()
        num_tweets = len(lines)
        for i, line in enumerate(lines):
            t_id, if_pos, tweet = line.strip().split(',')
            if_pos = int(if_pos)
            if if_pos:
                num_pos_tweets += 1
            else:
                num_neg_tweets += 1
            result, words, bigrams, trigrams = analyze_tweet(tweet)

            num_mentions += result['MENTIONS']
            max_mentions = max(max_mentions, result['MENTIONS'])
            num_pos_emojis += result['POS_EMOS']
            num_neg_emojis += result['NEG_EMOS']
            max_emojis = max(
                max_emojis, result['POS_EMOS'] + result['NEG_EMOS'])
            num_urls += result['URLS']
            max_urls = max(max_urls, result['URLS'])
            num_words += result['WORDS']
            min_words = min(min_words, result['WORDS'])
            max_words = max(max_words, result['WORDS'])
            all_words.extend(words)
           
            num_bigrams += result['BIGRAMS']
            num_Trigrams += result['TRIGRAMS']
            all_bigrams.extend(bigrams)
            all_Trigrams.extend(trigrams)

            write_status(i + 1, num_tweets)
    num_emojis = num_pos_emojis + num_neg_emojis
    
    unique_words = list(set(all_words))
    
    completeName = os.path.join('../features', train_data_file.split("-")[0] + '-unique.txt')
    with open(completeName , 'w') as uwf:
        uwf.write('\n'.join(unique_words))

    num_unique_words = len(unique_words)
    num_unique_bigrams = len(set(all_bigrams))
    
    num_unique_trigrams = len(set(all_Trigrams))
    print ('\nCalculating frequency distribution')
    # Unigrams
    freq_dist = FreqDist(all_words)
    

    completeName = os.path.join('../features', train_data_file.split("-")[0] + '-freqdist.pkl')
    with open(completeName, 'wb') as pkl_file:
        pickle.dump(freq_dist, pkl_file)
    
    print ('Saved uni-frequency distribution to %s') % completeName
    # Bigrams
    bigram_freq_dist = get_freqdist(all_bigrams)
    completeName = os.path.join('../features', train_data_file.split("-")[0] + '-freqdist-bi.pkl')
    with open(completeName, 'wb') as pkl_file:
        pickle.dump(bigram_freq_dist, pkl_file)

    print ('Saved bi-frequency distribution to %s') % completeName

    # Trigrams
    trigram_freq_dist = get_freqdist(all_Trigrams)
    completeName = os.path.join('../features', train_data_file.split("-")[0] + '-freqdist-Tri.pkl')

    with open(completeName, 'wb') as pkl_file:
        pickle.dump(trigram_freq_dist, pkl_file)

    print ('Saved Tri-frequency distribution to %s') % completeName

