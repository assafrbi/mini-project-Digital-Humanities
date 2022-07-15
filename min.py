import re
from pydoc import synopsis
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
import concurrent.futures
import pandas as pd
# import wikipedia
import numpy as np
import json
import glob
from imdb import Cinemagoer
import tqdm
import time
import nltk
# nltk.download('stopwords')
nltk.download('all')
# nltk.download('punkt')
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models
import collections
from operator import itemgetter
from scipy import stats
# Import the wordcloud library
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# %matplotlib inline
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


import swifter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Maximum number of threads that will be spawned
# MAX_THREADS = 6

# movie_title_arr = []
# movie_year_arr = []
# movie_genre_arr = []
# image_id_arr = []
# movie_synopsis_arr = []
# need_wiki_id = []
# need_wiki_name = []
# need_wiki_year = []


# ia = Cinemagoer()

# def getSynopsisFromImdb(id):
#   movie = ia.get_movie(id)
#   print("for id {}\n".format(id))
#   print("\n summary is :\n {}\n".format(movie.get('plot')[0]))
#   return movie.get('plot')[0]


# def getMovieTitle(header):
#     try:
#         return header[0].find("a").getText()
#     except:
#         return 'NA'

# def getReleaseYear(header):
#     try:
#         return header[0].find("span",  {"class": "lister-item-year text-muted unbold"}).getText().strip()[-5:-1]
#     except:
#         return 'NA'

# def getGenre(muted_text):
#     try:
#         return muted_text.find("span",  {"class":  "genre"}).getText().strip()
#     except:
#         return 'NA'
        
# def getsynopsys(movie, id, name, year):
#     try:
#         movie_synopsis = getSynopsisFromImdb(id)
#         return movie_synopsis
#     except:
#       # no imdb data -> need wiki
#       need_wiki_id.append(id)
#       need_wiki_name.append(name)
#       need_wiki_year.append(year)
#       return 'NA'

# def getImageId(image):
#     try:
#         return image.get('data-tconst')[2:]
#     except:
#         return 'NA'


# def main(imdb_url):
#     response = requests.get(imdb_url)
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Movie Name
#     movies_list  = soup.find_all("div", {"class": "lister-item mode-advanced"})
    
#     for indx, movie in enumerate(movies_list):
#         header = movie.find_all("h3", {"class":  "lister-item-header"})
#         muted_text = movie.find_all("p", {"class":  "text-muted"})[0]
#         imageDiv =  movie.find("div", {"class": "lister-item-image float-left"})
#         image = imageDiv.find("img", "loadlate")
        
#         # Movie Title
#         movie_title = getMovieTitle(header)
#         movie_title_arr.append(movie_title)

#         # Movie id
#         id = getImageId(image)
#         image_id_arr.append(id)
        
#         # Movie release year
#         year = getReleaseYear(header)
#         movie_year_arr.append(year)

#         # Movie Synopsys
#         synopsis = getsynopsys(movie, id, movie_title, year)
#         movie_synopsis_arr.append(synopsis)
        
#         # Genre
#         genre = getGenre(muted_text)
#         movie_genre_arr.append(genre)
        
# # An array to store all the URL that are being queried
# imageArr = []

# # Maximum number of pages one wants to iterate over
# MAX_PAGE =51

# # Loop to generate all the URLS.
# for i in range(1, 1500, 50):
#     imdb_url = f'https://www.imdb.com/search/title/?title_type=feature&countries=il&start={i}&ref_=adv_nxt'
#     imageArr.append(imdb_url)


# def download_stories(story_urls):
#     threads = min(MAX_THREADS, len(story_urls))
#     with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
#         time.sleep(0.3)
#         executor.map(main, story_urls)

# # Call the download function with the array of URLS called imageArr
# download_stories(imageArr)

# # Attach all the data to the pandas dataframe. You can optionally write it to a CSV file as well
# movieDf = pd.DataFrame({
#     "Title": movie_title_arr,
#     "Year": movie_year_arr,
#     "Genre": movie_genre_arr,
#     "Synopsis": movie_synopsis_arr,
#     "IMDb id": image_id_arr,
# })

# needWiki = pd.DataFrame({
#     "Title": need_wiki_name,
#     "Year": need_wiki_year,
#     "IMDb id": need_wiki_id,
# })

# movieDf.to_csv('israel_originated_movies_imdb.csv')
# needWiki.to_csv('movies_need_wiki.csv')



##### TOPIC MODELLING##########


summary_genre = ["Synopsis","Genre"]
data = pd.read_csv('israel_originated_movies_imdb.csv', usecols=summary_genre)
# print(data["Genre"].iloc[1])
# print("\n{}".format(data["Synopsis"].iloc[1]))
stopwords = stopwords.words("english")
# print (stopwords)
# print(data.isnull().sum())
data = data.dropna()
# print(data.isnull().sum())


data.to_csv('ready_for_topic_modeling.csv', index=False)

ready_data= pd.read_csv('ready_for_topic_modeling.csv', usecols=summary_genre)


def remove_new_line_characters(text):
    return re.sub('\s+', ' ', text)

def remove_quotes(text):
    text = re.sub("\'", "", text)
    text = re.sub('\"', '', text)
    return text

def lowercase(text):
    return text.lower()    

def remove_single_char_words(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def remove_verbs(text):
    final_text = ''
    allowedWordTypes = ["J","R","N"]
    tokens = nltk.word_tokenize(text)
    print(tokens)
    tagged = nltk.pos_tag(tokens)
    for w in tagged:
        if w[1][0] in allowedWordTypes:
            final_text += f' {w[0]}'
    return final_text

def text_preprocess(text):
    text = remove_new_line_characters(text)
    text = remove_quotes(text)
    text = lowercase(text)
    text = remove_single_char_words(text)
    text = remove_verbs(text)
    return text

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def lemmatization_x(sent, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc = nlp(" ".join(sent)) 
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

def join_list(the_list):
    str1 = " " 
    return str1.join(the_list)

def allocate_topics(i):
    return lda_model[corpus[i]][0]

def get_max_probability_topic(x, num_topics):
    return  max(x, key=itemgetter(1))[0]

def build_LDA_model(ready_data, source, num_topics):
    # Create the Dictionary and Corpus needed for Topic Modeling

    # Create Dictionary
    #id2word = corpora.Dictionary(data_lemmatized)
    id2word = corpora.Dictionary(ready_data[f'{source}_lemmatized'].tolist())

    # Create Corpus
    texts = ready_data[f'{source}_lemmatized'].tolist()

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    del texts

    # Build LDA model
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics, 
                                               random_state=7,
                                               eval_every=10,
                                               chunksize=100,
                                               passes=10,
                                               per_word_topics=True,
                                               workers=8)   
    return lda_model, corpus, id2word

def compute_metrics_per_topics(ready_data, source):
    # Calculate the frequency of words for this topic and for all corpora

    # Computed once for all corpus
    corpora_word_counts = [e.split(' ') for e in ready_data[f'{source}_preprocessed'].tolist()]
    corpora_word_counts = [item for sublist in corpora_word_counts for item in sublist]
    corpora_word_counts = collections.Counter(corpora_word_counts)
    corpora_word_counts_df = pd.DataFrame.from_dict(corpora_word_counts, orient='index')

    metrics = pd.DataFrame()
    for topic_n in ready_data[f'{source}_topic'].unique():
        topic_word_counts = [e.split(' ') for e in ready_data.loc[ready_data[f'{source}_topic']==topic_n][f'{source}_preprocessed'].tolist()]
        topic_word_counts = [item for sublist in topic_word_counts for item in sublist]
        topic_word_counts=collections.Counter(topic_word_counts)
        topic_word_counts_df = pd.DataFrame.from_dict(topic_word_counts, orient='index')

        word_counts_df = corpora_word_counts_df.merge(topic_word_counts_df, left_index=True, right_index=True, how='left').\
        rename({'0_x': 'corpora','0_y': f'{source}_topic'}, axis=1)
        word_counts_df = word_counts_df.fillna(0)

        metrics = metrics.append({'Topic': topic_n,
                                  'Probability': len(topic_word_counts)/len(corpora_word_counts_df),
                                  'Entropy': stats.entropy(pk=word_counts_df[f'{source}_topic'].values),
                                  'KL': stats.entropy(pk=word_counts_df[f'{source}_topic'].values, qk=word_counts_df['corpora'].values)
                   }, ignore_index=True) 

    return metrics.sort_values(by='Probability')    

def create_topic_specific_wordcloud(ready_data, source, topic):
    # Create a WordCloud object for specific topic
    temp = ready_data.loc[ready_data[f'{source}_topic']==0][f'{source}_preprocessed'].to_list()
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue',
                         width=1800, height=900)
    # Join the different processed titles together.
    long_string = ','.join(temp)
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    return wordcloud.to_image()


###### PRE-PROCESS #######

ready_data.loc[ready_data['Synopsis']!='']
source = 'Synopsis'
data = ready_data.copy()

print(f'---Doing pre-processing for {source}..')
print('Initial pre-processing..')


data[source] = data[source].apply(lambda x: text_preprocess(x))

data = data[source].to_list()
data_words = list(sent_to_words(data))


# Build the bigram model
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100, progress_per=200) # higher threshold fewer phrases.

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)

# Remove Stop Words
print('Removing stopwords..')
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
print('Creating bigrams..')
ready_data[f'{source}_words_bigrams'] = make_bigrams(data_words_nostops)
print('stage 2 done')
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python -m spacy download en_core_web_sm


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print('Lemmatizing words..')
ready_data[f'{source}_lemmatized'] = ready_data[f'{source}_words_bigrams'].swifter.apply(lambda x: lemmatization_x(x))
ready_data = ready_data.drop(columns=[f'{source}_words_bigrams'])

ready_data[f'{source}_preprocessed'] = ready_data[f'{source}_lemmatized'].apply(lambda x: join_list(x))

del bigram, bigram_mod, data_words, data, nlp
print('stage 3 done')



#### most common words####

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=2000, contour_width=3, contour_color='steelblue',
                     width=1400, height=600)
# Join the different processed titles together.
long_string = ','.join(ready_data[f'{source}_preprocessed'].to_list())
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()
print('stage 4 done')



###LDA MODEL####


num_topics = 6
lda_model, corpus, id2word = build_LDA_model(ready_data, source=source, num_topics=num_topics)

lda_model.print_topics()
print('stage 5 done')

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

ready_data['i'] = [i for i in range(len(ready_data))]
ready_data[f'{source}_topic'] = ready_data['i'].swifter.apply(lambda x: allocate_topics(x))
ready_data[f'{source}_topic'] = ready_data[f'{source}_topic'].apply(lambda x: get_max_probability_topic(x, num_topics))
