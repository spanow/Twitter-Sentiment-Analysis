import re
import sys
from pathlib import Path
import os
from nltk.stem import WordNetLemmatizer 
import nltk 
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



TRAIN_data=True
use_stemmer = False
use_lemmatizer = True



data_folder = Path("../dataset")
train_data = "training_data.csv"
test_data = "test_data.csv"
slang_words = "slang_words.csv"


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    #and greek characters
    word=word.decode('utf-8','ignore').encode('utf-8')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)

def write_status(i, total):
    ''' Writes status of a process to console '''
    sys.stdout.write('\r')
    sys.stdout.write('Processing %d/%d' % (i, total))
    sys.stdout.flush()


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
   
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)

    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)

    words = tweet.split()
    for word in words:
        if word in slang_words_dict:
            tweet = re.sub(word.encode('raw-unicode-escape'),slang_words_dict.get(word)+" ", tweet)

    
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)

    # delete stop words 3ayatni         
    words = tweet.split() 
           
    for word in words:
        if word in stop_words_dict:
            tweet = tweet.replace(" "+word+" ", " ")
            
            

    #replace hmmmmmm
    tweet = re.sub(r'h(m)+','',tweet)
    

    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)

    
    words = tweet.split()
    
    
   

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):

            if use_lemmatizer and use_stemmer:
                word = str(porter_lemmatizer.lemmatize(word))
                word = str(porter_stemmer.stem(word))
            elif use_stemmer:
                word = str(porter_stemmer.stem(word))
            elif use_lemmatizer:
               word = str(porter_lemmatizer.lemmatize(word)) 

            processed_tweet.append(word)

    return ' '.join(processed_tweet)

def read_slang_words(csv_file_name):
    slang_words_dict = {}
    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()
        total =len(lines)
        for i,line in enumerate(lines):
            
            word_to_replace = line[:line.find(',')]
            line = line[1 + line.find(','):]
            new_word = line[:line.find(',')]
            slang_words_dict[word_to_replace] = new_word
            write_status(i + 1, total)
    return slang_words_dict
            



def preprocess_csv(csv_file_name, processed_file_name, test_file=False,head=True):

    completeName = os.path.join('../clean_data', processed_file_name)
    save_to_file = open(completeName, 'w')
    

    with open(csv_file_name, 'r') as csv:
        if head :
            lines = csv.readline()
            head = False
        lines = csv.readlines()
        total = len(lines)
        
        for i, line in enumerate(lines):
            
            tweet_id = line[:line.find(',')]
            
            if not test_file:
                line = line[1 + line.find(','):]
                
                positive = int(line[:line.find(',')])

            line = line[1 + line.find(','):]
            tweet = line
            processed_tweet = preprocess_tweet(tweet)
            if processed_tweet=="": 
                continue
            if not test_file:
                save_to_file.write('%s,%d,%s\n' %
                                   (tweet_id, positive, processed_tweet))
            else:
                save_to_file.write('%s,%s\n' %
                                   (tweet_id, processed_tweet))
            write_status(i + 1, total)
    save_to_file.close()
    print ('\n Saved processed tweets to: %s' % processed_file_name)
    return processed_file_name


if __name__ == '__main__':
    
    
  


    train_data_path = data_folder / train_data
    test_data_path = data_folder / test_data
    slang_words_path = data_folder / slang_words


    slang_words_dict = read_slang_words(str(slang_words_path)) 
    stop_words_dict = set(stopwords.words('english'))
    


    if TRAIN_data:
        
        which_data = train_data.split("_")[0]
        processed_file_name = which_data + '-processed.csv'
        if use_lemmatizer and use_stemmer:
            porter_lemmatizer = WordNetLemmatizer() 
            porter_stemmer = PorterStemmer()
            processed_file_name = which_data + '-processed-stemmed-lemmatized.csv'
        elif use_lemmatizer :
            porter_lemmatizer = WordNetLemmatizer() 
            processed_file_name = which_data + '-processed-lemmatized.csv'
        elif use_stemmer:
            porter_stemmer = PorterStemmer()
            processed_file_name = which_data + '-processed-stemmed.csv'
        preprocess_csv(str(train_data_path), processed_file_name, test_file=False ,head=True)
    else:
    
        which_data = test_data.split("_")[0]
    
        processed_file_name = which_data + '-processed.csv'

    

        if use_lemmatizer and use_stemmer:
            porter_lemmatizer = WordNetLemmatizer() 
            porter_stemmer = PorterStemmer()
            processed_file_name = which_data + '-processed-stemmed-lemmatized.csv'
        elif use_lemmatizer :
            porter_lemmatizer = WordNetLemmatizer() 
            processed_file_name = which_data + '-processed-lemmatized.csv'
        elif use_stemmer:
            porter_stemmer = PorterStemmer()
            processed_file_name = which_data + '-processed-stemmed.csv'
        preprocess_csv(str(test_data_path), processed_file_name, test_file=True ,head=True)
   

