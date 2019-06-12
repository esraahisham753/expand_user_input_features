###############################################
#                                             #
# download stopwords 34an my3mlsh error ta7t  #
#                                             #
###############################################

import nltk
import re
import nltk as n
from nltk.stem import *
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import ISRIStemmer
import xlrd
import pickle
import numpy as np
import tflearn as tf
from tensorflow.python.framework import ops
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

#########################
#                       #
#      Functions        #
#                       #
#########################

def preprocessing_test_data(user_string):
    tokens_array = n.tokenize._treebank_word_tokenizer.tokenize(user_string)
    stemmer = ISRIStemmer()
    stop_words = stopwords.words('arabic')
    preprocessing_result_question = list()
    for word in tokens_array:
        word = stemmer.norm(word, num=1)      # remove diacritics which representing Arabic short vowels
        if not word in stop_words:           # exclude stop words from being processed
          word = stemmer.pre32(word)        # remove length three and length two prefixes in this order
          word = stemmer.suf32(word)        # remove length three and length two suffixes in this order
          word = stemmer.waw(word)          # remove connective ??? if it precedes a word beginning with ???
          word = stemmer.norm(word, num=2)  # normalize initial hamza to bare alif
          preprocessing_result_question.append(word)
    return preprocessing_result_question

################################
#### user input features########
################################
def user_features(preprocessed_data):
    features_names = 0
    with open('Question_features_names.tmp', 'rb') as dic:
        features_names = pickle.load(dic)
    # print(features_names)
    # print(len(Q_vectorizer.get_feature_names()))
    #pr = preprocessing_test_data(preprocessed_data)
    QU = ''
    for i in range(len(preprocessed_data)):
        QU += preprocessed_data[i]
        QU += ' '
    # print(QU)
    QU_list = [QU]
    QU_vectorizer = TfidfVectorizer()
    QU_vectorizer.fit(QU_list)
    # print(QU_vectorizer.get_feature_names())
    # print(QU_vectorizer.idf_)
    UQuestion_features = QU_vectorizer.transform(QU_list)
    # print(UQuestion_features[0, 1])
    # print(Q_vectorizer.get_feature_names())
    QU_features = np.zeros((1, 357))
    for j in range(357):
        for k in range(len(QU_vectorizer.get_feature_names())):
            if (QU_vectorizer.get_feature_names()[k] == features_names[j]):
                QU_features[0, j] = UQuestion_features[0, k]
    print(QU_features.shape)
    return QU_features



def process_user_input(user_string):
    preprocessed_string = preprocessing_test_data(user_string)
    features = user_features(preprocessed_string)
    ops.reset_default_graph()
    net = tf.input_data(shape=[None, 357], name='input')
    net = tf.fully_connected(net, 256, activation='relu')
    net = tf.fully_connected(net, 128, activation='relu')
    net = tf.fully_connected(net, 64, activation='relu')
    output = tf.fully_connected(net, 277, activation='softmax')
    output = tf.regression(output, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='output')
    # define model
    model = tf.DNN(output)
    if (os.path.exists('model.tfl.meta')):
        model.load('model.tfl')
        #print("exists")
    else:
        print("does not exist")
    prediction = model.predict(features)
    #print(prediction)
    predictions_list = []
    for i in range(len(prediction)):
        predictions_list.append(np.argmax(prediction[i]) + 1)
    #print(predictions_list)
    classes = 0
    with open('classes.tmp', 'rb') as dic:
        classes = pickle.load(dic)
    return classes[predictions_list[0]]

