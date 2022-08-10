#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:36:44 2021

@author: russell
"""

import numpy as np
import re
from pandas import read_excel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from SupervisedAlgorithm import Logistic_Regression_Classifier
from SupervisedAlgorithm import SVM_Classifier
from SupervisedAlgorithm import SGD_Classifier
from SupervisedAlgorithm import KNN_Classifier
from SupervisedAlgorithm import RandomForest_Classifier
from SupervisedAlgorithm  import MultinomialNB_Classifier
from SupervisedAlgorithm  import ExtraTree_Classifier
from sklearn.feature_extraction.text import TfidfVectorizer
   

#Append Path If particular Library is not found
import sys
sys.path.append('/Users/sazzed/eman2-sphire-sparx/lib/python3.7/site-packages/')

from SupervisedAlgorithm import  Performance


BingLiu_lexicon = {}
dic_sentic_net = {}
vader_lexicon = {}
sentic_net = {}

unique_words = {}


num_of_words = -1


def draw_scatter_plot(x, y):

    import numpy as np
    import matplotlib.pyplot as plt
    
    N = len(x)
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()



def write_to_CSV(filename, X_data, Y_label):
    
    import csv
    
    #c = list(zip(X_data, Y_label))
    #random.shuffle(c)
    #X_data, Y_label  = zip(*c)
    
    #persons=[('Lata',22,45),('Anil',21,56),('John',20,60)]
    ##csvfile=open("BanglaResaurant.csv",'w', newline='')
    ##csvfile=open("YelpResaurant.csv",'w', newline='')
    csvfile=open(filename,'w', newline='')
    
    obj=csv.writer(csvfile)
    
    data= []
    label = []
    
    for i in range(len(X_data)):
        data.append(X_data[i].strip("\n"))
        label.append(Y_label[i])
    
    for element in zip(data, label):
        obj.writerow(element)
    
    csvfile.close()
    
    
def read_data_from_excel(excel_file, sheet_name):
    from pandas import read_excel
    sheet_name = sheet_name.strip()
    data = read_excel(excel_file, sheet_name = sheet_name)
    #data = pd.read_csv("/Users/russell/Documents/NLPPaper/Comments.csv") #text in column 1, classifier in column 2.
    numpy_array = data.values
   # print(numpy_array)
    X = numpy_array[:,0]
    Y = numpy_array[:,1]
   
    
    #print(len(X))
    
    Y = Y.astype('int')

    return X, Y
    
    

def write_to_text_file(data, filename):
    f = open(filename, "w+")
    for i in range(len(data)):
        #print(data[i])
        f.write(str(data[i]) + "\n")
        
        
def read_text_data_independent_set(filepath):
     reviews = []
     with open(filepath) as file:
         for line in file:
             
             line = line.strip()
             if (len(line) < 5):
                 continue
             reviews.append(line)
             # print("*****",line)
             #line = line.strip()
             #tokens = line.split(",")
             #fTokens = []
          
     print("Number of review: ",len(reviews))
     return reviews
        
        
def read_feature_data_from_textfile(filepath):
     
    data = []
    
    #selected_features = [1,0,0,0,0,0,0,0,0,0,0]
    
    #selected_features = [0,0,0,0,0,0,1,1,1,1,1]
    
    selected_features = [1,1,1,1,1,1,1,1,1,1,1]
    #selected_features = [1,1,1,0,0,0,0,0,0,0,0]
    #selected_features = [0,0,0,1,1,1,1,1,1,0,0]
    #selected_features = [0,0,0,0,0,0,0,0,0,1,1]
    
    with open(filepath) as file:
        for line in file:
            line = line.replace(",,",",")
           # print("*****",line)
            line = line.strip()
            tokens = line.split(",")
            fTokens = []
            
            
            #for t in tokens:
            for index in range(len(tokens)):
                if selected_features[index]== 1 :
                   # print("----",index, float(tokens[index]))
                    fTokens.append(float(tokens[index]))
                    #
            
            data.append(fTokens)

    return data

#-----------Clause --------
def compute_dependent_clause_ratio(reviews):
    
    file = open('/Users/sazzed/Documents/Other/Research/Restaurant/subordinate.txt', 'r') 
    
    clause = {}
    for line in file: 
        tokens = line.split(",")
        for token in tokens:
            if (len(token.strip()) > 1):
                clause[token.strip()] = 1
    
    #for key in clause:
       # print(key)
    
    #print(len(clause))
    
    in_clause= 0
    out_clause= 0
    
    clause_reviews = []
    for review in reviews:
        num_of_clause = 0
        tokens = review.split()
        for token in tokens: 
            if token in clause:
                in_clause += 1
                num_of_clause += 1
            else:
                out_clause += 1
    
        clause_reviews.append(num_of_clause) 
        
    print(in_clause, out_clause, in_clause + out_clause )
    
    
    print("SUM: ",sum(clause_reviews),in_clause * 100/ out_clause)
    
    
    return  in_clause, out_clause
    
    
    write_to_text_file(clause_reviews, "/Users/russell/Downloads/clause_bangla.txt" )

def sentiment_dictionary_BingLiu():
    file = open('/Users/sazzed/Documents/Other/Research/resource/positive.txt', 'r') 
    for line in file: 
        token = line.split()
        key = ''.join(token)
        BingLiu_lexicon[key] = 1
        
    file = open('/Users/sazzed/Documents/Other/Research/resource/negative.txt', 'r') 
    for line in file: 
        token = line.split()
        key = ''.join(token)
        BingLiu_lexicon[key] = -1
        

def get_vader_lexicon():
    file = open('/Users/sazzed/Documents/Other/Research/Restaurant/vader_lexicon.txt', 'r') 
    sentiments = []
    for line in file: 
        token = line.split("\t")
        key = token[0]
        score = token[1]
        #print(key, score)
        vader_lexicon[key] = float(score)
        sentiments.append(key + "," + score)
        
def senticnet_lexicon():
   
    #file = open('/Users/russell/Downloads/senticnet-4.0/senticnet4.txt', 'r') 
    file = open('/Users/russell/Downloads/senticnet-5.0/senticnet5.txt', 'r') 
    for line in file: 
         #elements = line.rstrip().split(' ')[3:]
        (key, val,score) = line.split()
        sentic_net[key] = score
        
        

#--------------- Extract Word-----
def extract_word_from_reviews(reviews):
    
    docs = reviews
    
    vectorizer = CountVectorizer(ngram_range=(1,1), tokenizer=lambda x: x.split())
# tokenize and build vocab
    vectorizer.fit(docs)
    #print('vocabulary: ', vectorizer.vocabulary_)
    
    bag_of_words = vectorizer.transform(docs)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    words_freq = words_freq[:]

    i = 0
   
    
    '''
    
    for word,freq  in words_freq:
        if freq > 100:
            print(i, word, freq)
        else:
            break
        i += 1
    ''' 
        

   
    print( words_freq[:20])
    
         
        
#--------- Get Data -------
def get_data_n_label_from_yelp_review(csv_file, sheet_name):
   
    #data = read_excel('/Users/russell/Documents/NLPPaper/Aspect_Sentiment_Analysis/Dataset/Food-Review-2.xlsx', sheet_name = my_sheet_name)
    data = read_excel(csv_file, sheet_name = sheet_name)
    #data = pd.read_csv("/Users/russell/Documents/NLPPaper/Comments.csv") #text in column 1, classifier in column 2.
    numpy_array = data.values
   # print(numpy_array)
    X = numpy_array[2:,4]
    Y = numpy_array[2:,3]
   # print(X[100:200])s
    #Z = numpy_array[:,2]
    
    #for i in X:
     #   print("-------", i)
    reviews = []
    ratings = []
    
    neg = 0
    pos = 0
    
    for i  in range(len(X)):
        #print( user_review)
        if len(str(X[i])) < 5 or int(Y[i]) == 3:
           continue
       
        if (int(Y[i]) > 3):
            ratings.append('1')
            pos += 1
            
        if (int(Y[i]) < 3):
            ratings.append('0')
            neg += 1
        
        reviews.append(X[i])
        #ratings.append(Y[i])
        
    print("Yelp Reviews: ", neg, pos)
    return np.array(reviews), np.array(ratings)


##def get_data_n_label_from_yelp_review_1702_pos_613_neg(csv_file, sheet_name, neg_count, pos_count):
def get_data_n_label_from_yelp_review_X_pos_Y_neg(csv_file, sheet_name, neg_count, pos_count):

    #data = read_excel('/Users/russell/Documents/NLPPaper/Aspect_Sentiment_Analysis/Dataset/Food-Review-2.xlsx', sheet_name = my_sheet_name)
    data = read_excel(csv_file, sheet_name = sheet_name)
    #data = pd.read_csv("/Users/russell/Documents/NLPPaper/Comments.csv") #text in column 1, classifier in column 2.
    numpy_array = data.values
   # print(numpy_array)
    X = numpy_array[2:,4]
    Y = numpy_array[2:,3]
   # print(X[100:200])s
    #Z = numpy_array[:,2]
    
    
    from sklearn.utils import shuffle

    X, Y = shuffle(X, Y)
    
    #for i in X:
     #   print("-------", i)
    pos_reviews = []
    #pos_ratings = []
    
    neg_reviews = []
    #neg_ratings = []
    
    for i  in range(len(X)):
        #print( user_review)
        if len(str(X[i])) < 5 or int(Y[i]) == 3:
           continue
       
        if (int(Y[i]) > 3 and len(pos_reviews) < pos_count):
            #ratings.append('1')
            pos_reviews.append(X[i])
            
        if (int(Y[i]) < 3 and len(neg_reviews) < neg_count):
            #ratings.append('0')
            neg_reviews.append(X[i])
        
        #reviews.append(X[i])
        #ratings.append(Y[i])
        
    reviews = neg_reviews + pos_reviews  #[:11838]
    ratings = len(neg_reviews) * [0] +  len(pos_reviews) * [1] 

    print("Neg: ", len(neg_reviews))
    print("Pos: ", len(pos_reviews))
        
    return np.array(reviews), np.array(ratings)


def get_data_n_label_from_excel(excel_file, sheet_name, label_column = 2):
   
    data = read_excel(excel_file, sheet_name = sheet_name)
    #data = pd.read_csv("/Users/russell/Documents/NLPPaper/Comments.csv") #text in column 1, classifier in column 2.
    numpy_array = data.values
   # print(numpy_array)
    X = numpy_array[1:,0]
    Y = numpy_array[1: ,label_column]
    
    
  
    print("------>>>>> ", len(X))
    return X, Y



def getKaggleRestaurantReviews():
     # Kaggle Restaurant- Other
    X, Y = get_data_n_label_from_excel("/Users/russell/Documents/NLP/Restaurant/Data/KaggleBangla-2685.xlsx", 'KaggleBangla-2685', 1)
    
   
    #print(Y)
    return  X, Y 
   


def getKaggleRestaurantReviews_2700():
     # Kaggle Restaurant- Other
    X, Y = get_data_n_label_from_excel("/Users/russell/Documents/NLP/Restaurant/Data/15000_restaurant_reviews_fixed.xlsx", '15000_restaurant_reviews', 1)
    
    pos_reviews = []
    #pos_ratings = []
    
    neg_reviews = []
    #neg_ratings = []
    
    for i  in range(len(X)):
        #print( user_review)
        if len(str(X[i])) < 5 or int(Y[i]) == 3:
           continue
       
        if (int(Y[i]) > 3 and len(pos_reviews) < 700):
            #ratings.append('1')
            pos_reviews.append(X[i])
            
        if (int(Y[i]) < 3 and len(neg_reviews) < 2000):
            #ratings.append('0')
            neg_reviews.append(X[i])
        

    reviews = neg_reviews + pos_reviews  #[:11838]
    ratings = len(neg_reviews) * [0] +  len(pos_reviews) * [1] 

    print("Neg: ", len(neg_reviews))
    print("Pos: ", len(pos_reviews))
        
    write_to_CSV("/Users/russell/Documents/NLP/Restaurant/Data/KaggleBangla-2685.csv", reviews, ratings)
    return np.array(reviews), np.array(ratings)
    #print(Y)
    #return  X, Y 

def get_ELP_data_four_classes():
     # Kaggle Restaurant- Other
    #X, Y = get_data_n_label_from_excel("/Users/russell/Documents/NLP/Restaurant/Data/15000_restaurant_reviews_fixed.xlsx", '15000_restaurant_reviews', 1)
    
    #country = "Bangladesh"#"China"#"kenya" #"Finland" #  "mayanmar" #
    X_data_0 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Finland.txt")
    X_data_1 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Kenya.txt")
    X_data_2 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/China.txt")
    X_data_3 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Bangladesh.txt")
   # X_data_4 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Mayanmar.txt")
   
    #X_data_0 =  X_data_0[:60]
    #X_data_1 =  X_data_1[:60]
    #X_data_2 =  X_data_2[:60]
    #X_data_3 =  X_data_3[:60]
    #X_data_4 =  X_data_4[:60]
    
    
    print("X_data_0: ", len(X_data_0))
    print("X_data_1: ", len(X_data_1))
    print("X_data_2: ", len(X_data_2))
    print("X_data_3: ", len(X_data_3))

    reviews = X_data_0 + X_data_1 + X_data_2 + X_data_3 
    ratings = len(X_data_0) * [0] +  len(X_data_1) * [1] +  len(X_data_2) * [2] +  len(X_data_3) * [3] 

    reviews = get_data_named_entity_removed(reviews)
    #write_to_CSV("/Users/russell/Documents/NLP/Restaurant/Data/KaggleBangla-2685.csv", reviews, ratings)
    return np.array(reviews), np.array(ratings)
    #print(Y)
    #return  X, Y 
   
    
    
def get_ELP_data():
     # Kaggle Restaurant- Other
    #X, Y = get_data_n_label_from_excel("/Users/russell/Documents/NLP/Restaurant/Data/15000_restaurant_reviews_fixed.xlsx", '15000_restaurant_reviews', 1)
    
    #country = "Bangladesh"#"China"#"kenya" #"Finland" #  "mayanmar" #
    X_data_0 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Finland.txt")
    X_data_1 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Kenya.txt")
    X_data_2 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/China.txt")
    X_data_3 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Bangladesh.txt")
    X_data_4 = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Mayanmar.txt")
   
    #X_data_0 =  X_data_0[:60]
    #X_data_1 =  X_data_1[:60]
    #X_data_2 =  X_data_2[:60]
    #X_data_3 =  X_data_3[:60]
    #X_data_4 =  X_data_4[:60]

    reviews = X_data_0 + X_data_1 + X_data_2 + X_data_3 + X_data_4
    ratings = len(X_data_0) * [0] +  len(X_data_1) * [1] +  len(X_data_2) * [2] +  len(X_data_3) * [3] + +  len(X_data_4) * [4]  

    #write_to_CSV("/Users/russell/Documents/NLP/Restaurant/Data/KaggleBangla-2685.csv", reviews, ratings)
    return np.array(reviews), np.array(ratings)
    #print(Y)
    #return  X, Y 
   
    

def getBanglaRestaurantReviews():


    # Bangla Restaurant- Salim
    X, Y = get_data_n_label_from_excel("/Users/russell/Documents/NLP/Restaurant/Data/BanglaResaurant.xlsx", 'BanglaResaurant', 1)
    
    
    #Bangladdeshi restaurant
    
    csv_filename_0 = '/Users/russell/Documents/NLP/Aspect_Sentiment_Analysis/Dataset/Food-Review-2.xlsx'
    csv_filename_1 = '/Users/russell/Documents/NLP/Aspect_Sentiment_Analysis/Dataset/Food-Review-3.xlsx'
    csv_filename_2 = '/Users/russell/Documents/NLP/Aspect_Sentiment_Analysis/Dataset/Food-Review-4.xlsx'
    
    csv_filename_3 = '/Users/russell/Downloads/restaurant-2.xlsx'
    csv_filename_4 = '/Users/russell/Downloads/rudro.xlsx'
     


    X_1, Y_1 = get_data_n_label_from_excel(csv_filename_0, 'Rio Cafe')
    X_2, Y_2 = get_data_n_label_from_excel(csv_filename_0,'Royel Buffet')
    X_3, Y_3 = get_data_n_label_from_excel(csv_filename_0,'Absolute BBQ BD ')
    X_4, Y_4 = get_data_n_label_from_excel(csv_filename_0,'Pizza Inn Bangladesh')
    X_5, Y_5 = get_data_n_label_from_excel(csv_filename_1,'Sbarro Bangladesh')
    X_6, Y_6 = get_data_n_label_from_excel(csv_filename_1,'Lakeshore Banani')
    X_7, Y_7 = get_data_n_label_from_excel(csv_filename_1,'Flavours Music Cafe')
    X_8, Y_8 = get_data_n_label_from_excel(csv_filename_1,'The Pit Grill')
    X_9, Y_9 = get_data_n_label_from_excel(csv_filename_2,'Sheet1')
    X_10, Y_10 = get_data_n_label_from_excel(csv_filename_2,'Woodhouse Grill')
    X_11, Y_11 = get_data_n_label_from_excel(csv_filename_2,'Baton Rough Restaurant')
    X_12, Y_12 = get_data_n_label_from_excel(csv_filename_2,'Tree House')
    X_13, Y_13 = get_data_n_label_from_excel(csv_filename_2,'Cafe Entro Banani')
    X_14, Y_14 = get_data_n_label_from_excel(csv_filename_2,'Spicy Ramna')

    X_15,Y_15 = get_data_n_label_from_excel(csv_filename_3,'Sheet1')
    X_16,Y_16 = get_data_n_label_from_excel(csv_filename_4,'Sheet1')
    
    X_data = np.concatenate((X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8,X_9, X_10, X_11, X_12, X_13, X_14, X_15, X_16), axis=None)
    Y_label = np.concatenate((Y_1, Y_2, Y_3, Y_4,Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13, Y_14, Y_15, Y_16), axis=None)
    
    #Y_test = Y_label.astype('int')
    print("Total : ", len(X_data))
    
    return X_data, Y_label




    
#---------- Coverage ------

def findLexiconCoverage(reviews):
    
    sentiment_dictionary_BingLiu()
    get_vader_lexicon()
    #BingLiu_lexicon =  vader_lexicon
    in_lexicon = 0
    not_in_lexicon = 0
    for i in range(len(reviews)):

       review = reviews[i]
       review = review.replace("�","")
       
       tokens = review.split()
       
       for token in tokens:
           if token in BingLiu_lexicon:
               in_lexicon += 1
               #print(token)
           else:
               not_in_lexicon += 1
       
     
    print("Covearge: ", in_lexicon, in_lexicon + not_in_lexicon )
    print("Covearge: ", in_lexicon / (in_lexicon + not_in_lexicon) )
    
    
def getLexiconCoverageOfBingLiu(reviews):
    
    sentiment_dictionary_BingLiu()
   
    #BingLiu_lexicon = vader_lexicon
    in_lexicon = []
    not_in_lexicon = 0
    for i in range(len(reviews)):

       review = reviews[i]
       review = review.replace("�","")
       
       tokens = review.split()
       
       for token in tokens:
           if token in BingLiu_lexicon:
               #in_lexicon += 1
               in_lexicon.append(token)
               #print(token)
           else:
               not_in_lexicon += 1
       
     
    ##print("Covearge: ", in_lexicon, in_lexicon + not_in_lexicon )
    #print("Covearge: ", in_lexicon / (in_lexicon + not_in_lexicon))
    
    print("lex: ", in_lexicon)
    total = len(in_lexicon) + not_in_lexicon
    r = len(in_lexicon) / total
    return r


def getLexiconCoverageOfVADER(reviews):
    
    
    in_lexicon = 0
    not_in_lexicon = 0
    for i in range(len(reviews)):

       review = reviews[i]
       review = review.replace("�","")
       
       tokens = review.split()
       
       for token in tokens:
           if token in vader_lexicon:
               in_lexicon += 1
               #print(token)
           else:
               not_in_lexicon += 1
       
     
    ##print("Covearge: ", in_lexicon, in_lexicon + not_in_lexicon )
    #print("Covearge: ", in_lexicon / (in_lexicon + not_in_lexicon))
    
    ratio =  in_lexicon/ (in_lexicon + not_in_lexicon)
    return ratio
    
#--------Flesch_reading_ease------
def calculate_num_of_syllables(review):
    
    import syllapy
    
    tokens = review.split()
    
    total_syllables = 0
    for token in tokens:
        count = syllapy.count(token)
        ##print(token, " : " ,count)
        total_syllables += count
    
    
    ##for i in range(len(reviews)):

    return total_syllables
   

def get_num_of_unique_words(reviews):
    
    
    from collections import Counter
    
    word_list = []
   
    for i in range(len(reviews)):

        review = reviews[i]
        review = review.replace("�","")
        review = review.replace("·",".")
        split_sentences  = re.split('[.!]', review) # re.split('[^a-zA-Z][]', review)
     
       #print("---Rev----: \n", review)
       #print("\n")
        num_of_sentences = 0
      
        for s in split_sentences:
          
           if len(s) > 1:
              #print("Sen: ", s)
              num_of_sentences += 1
              tokens = s.split()
              #print(tokens)
              for token in tokens:
                  token = re.sub(r'[^a-zA-Z]', '', token)
                  if (len(token) < 2):
                      continue
                  unique_words[token] = 1
                  word_list.append(token)
         
    print("#Num of unique words: ", len(unique_words))
    
    x = Counter(word_list)
    x = x.most_common()[:200]
    #print(x)
    
    
    
def get_num_of_words_n_sentences(review):
     
   
   review = review.replace("�","")
   review = review.replace("·",".")
   split_sentences  = re.split('[.!]', review) # re.split('[^a-zA-Z][]', review)
 
   #print("---Rev----: \n", review)
   #print("\n")
   num_of_sentences = 0
   num_of_words = 0
   for s in split_sentences:
      
       if len(s) > 1:
          #print("Sen: ", s)
          num_of_sentences += 1
          tokens = s.split()
          #print(tokens)
          num_of_words += len(tokens)

   return num_of_words, num_of_sentences
      

def Flesch_reading_ease(reviews):
    
    import statistics
    from statistics import mean
    ##count = syllapy.count('additional')
    #print(count)
    
    flesch = []
    for i in range(len(reviews)):
       
        review = reviews[i]
        
        tokens = review.split()
        
        #if len(tokens) < 50:
           # continue
        
        total_syllables = calculate_num_of_syllables(review)
        
        total_words, total_sen = get_num_of_words_n_sentences(review)
        
        print("----",total_words, total_sen)
        
        term1 = 206.835
        term2 = 1.015 * float(total_words/total_sen)
        term3 = 84.6 * float(total_syllables/total_words)
        
        
    
        score = term1 - term2 - term3
        
        if score < 0 or score > 100:
            continue

        print(review)
        print("Score: ", i, "---> ",score)
        flesch.append(score)
        #break
    
    print("Length: ", len(flesch))
    print("AVerage:", mean(flesch))
    print("STD:", statistics.stdev(flesch))
    
def Flesch_Kincaid_Grade_Level(reviews):
    
    import statistics
    from statistics import mean
    ##count = syllapy.count('additional')
    #print(count)
    
    flesch = []
    for i in range(len(reviews)):
       
        review = reviews[i]
        
        #tokens = review.split()
        
        #if len(tokens) < 50:
           # continue
        
        total_syllables = calculate_num_of_syllables(review)
        
        total_words, total_sen = get_num_of_words_n_sentences(review)
        
        print("----",total_words, total_sen)
        
        ASL = float(total_words/total_sen)
        ASW = float(total_syllables/total_words)
        
        FKRA = (ASL * 0.39) + (11.8 * ASW) - 15.59
        
        
        if FKRA < 0 or FKRA > 100:
            continue

        print(review)
        print("Score: ", i, "---> ",FKRA)
        flesch.append(FKRA)
        #break
    
    print("Length: ", len(flesch))
    print("AVerage:", mean(flesch))
    print("STD:", statistics.stdev(flesch))
    
  
def Coleman_Liau_index(reviews):
    
    import statistics
    from statistics import mean
    ##count = syllapy.count('additional')
    #print(count)
    
    flesch = []
    for i in range(len(reviews)):
       
        review = reviews[i]
        total_syllables = calculate_num_of_syllables(review)
        
        total_words, total_sen = get_num_of_words_n_sentences(review)
        term1 = 0.0588 
        term2 = 0.296 * S * float(total_words/total_sen)
        term3 = 15.8
        
        
    
        score = term1 - term2 - term3
        
        if score < 0 or score > 100:
            continue

        print(review)
        print("Score: ", i, "---> ",score)
        flesch.append(score)
        #break
    
    print("Length: ", len(flesch))
    print("AVerage:", mean(flesch))
    print("STD:", statistics.stdev(flesch))
    
    
#----- 
def mannwhitneyu(dataset_1, dataset_2):
    import numpy as np
    from scipy.stats import norm
    
    _, pnorm = mannwhitneyu(dataset_1, dataset_2, use_continuity=False,
                        method="asymptotic")
    
    print(pnorm)
    
    
#----- Evaluate Independent Set ----
    
def evaluate_indepenent_set():
    ##independent_reviews = read_text_data_independent_set("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/Finland.txt")
    
    independent_reviews = read_text_data_independent_set("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/China.txt")
    
    
    print(independent_reviews[2])
    data_11, label_11 = getBanglaRestaurantReviews()
    data_12, label_12 = getKaggleRestaurantReviews()
    
    
    print("Length of BanglaRestaurant  ", len(data_11))
     
    print("Length of KaggleRestaurant Bangladesh ", len(data_12))
    print("Total: ", len(data_11) + len(data_12))
    
    data_1 = np.concatenate((data_11, data_12), axis=0)
    ##label_1 = np.concatenate((label_11, label_12), axis=0)
    
    # data_1, _ = getBanglaRestaurantReviews()
    data_2, label_2 = get_data_n_label_from_yelp_review_X_pos_Y_neg("/Users/russell/Downloads/yelp.xlsx", "yelp",1676, 3311)
   
    #data_2, label_2 = get_data_n_label_from_yelp_review_X_pos_Y_neg("/Users/russell/Downloads/yelp.xlsx", "yelp",2579, 3500)
   
    print("Length of data-2 ", len(data_2))
    #data_1 = data_1[:]
    data_2 = data_2[:len(data_1)] ##data_2[:5000] 
    #get_class_distribution(label_2)
    
    num_of_dependent_reviews = len(data_1) + len(data_2)
    #write_to_CSV("/Users/russell/Documents/NLP/Restaurant/Data/Yelp_4987.csv", data_2, label_2)
   
    data = np.concatenate((data_2, data_1, independent_reviews, data_2[0:len(independent_reviews)]), axis=0)
    #data = data_1 + data_2
    
    label = [1] * len(data_2) + [0] *  len(data_1) + [0] *  len(independent_reviews) + [1] *  len(independent_reviews)
    
      
    
    label = np.asarray(label)
    
    data = np.asarray(data)
    
    
    data = get_tfidf(data)
    
    
    
    
    classifier = Logistic_Regression_Classifier() 
        
        #classifier = Ridge_Regression_Classifier()
        
        
        #classifier = RandomForest_Classifier()
        
    ##classifier = SVM_Classifier()
        #classifier = KNN_Classifier()
        
        #classifier = SGD_Classifier()
        
        
        #classifier = LDA_Classifier()
        
        #classifier = MultinomialNB_Classifier()
        
    
    data_train = data[:2 * -len(independent_reviews)]
    label_train = label[:2 * -len(independent_reviews)]
    
    label_test = label[num_of_dependent_reviews:]
    data_test = data[num_of_dependent_reviews:]
    

    prediction = classifier.predict(data_train, label_train, data_test)

    performance = Performance() 


    conf_matrix,precision,  recall, f1_score, acc = performance.get_results(label_test, prediction)

    
    print(conf_matrix)
    #print



    
#----- Get Statistical Attributes ----
def get_common_words():
    
    file = open('/Users/russell/Documents/NLP/Restaurant/3000_common_words.txt', 'r') 
    common_words = {}
    for line in file: 
        #print(line)
        token = line.split()
        
        token[0] = token[0].strip()
        ##print(token[0])
        common_words[token[0]] = 1
        
    print("Common: ", len(common_words))
    count = 0
    for unique_word in unique_words:   
        if unique_word not in common_words:
            count += 1
            #print(unique_word)
            
    print("Non common word in corpus", count)
        
 
def get_negation_words(reviews):
    file = open('/Users/sazzed/Documents/Other/Research/Restaurant/negation.txt', 'r') 
    negation_words = {}
    for line in file: 
        #print(line)
        tokens = line.split(",")
        
        for token in tokens: 
            negation_words[token.strip()] = 1
        
    #for negation in negation_words:
      #  print(negation)
        
        
    negation_word = 0
        
    for i in range(len(reviews)):

       review = reviews[i]
       review = review.replace("�"," ")
       split_sentences  = re.split('[.!]', review) # re.split('[^a-zA-Z][]', review)
     

       for s in split_sentences:
           if len(s) > 1:
              tokens = s.split()
              for token in tokens:
                  if token in negation_words:
                      negation_word += 1
                      
    print("negation_word: ", negation_word) # "; Percent: ", round((negation_word/num_of_words * 100),2))
    return negation_word
    
    
def get_preposition(reviews):
    
    file = open('/Users/sazzed/Documents/Other/Research/Restaurant/preposition.txt', 'r') 
    negation_words = {}
    for line in file: 
        #print(line)
        tokens = line.split()
        
        if len(tokens) < 3:
            continue
        
        #print(tokens[0].strip())
        negation_words[tokens[0].strip()] = 1
        
    preposition_word = 0
        
    for i in range(len(reviews)):

       review = reviews[i]
       review = review.replace("�"," ")
       split_sentences  = re.split('[.!]', review) # re.split('[^a-zA-Z][]', review)
     

       for s in split_sentences:
           if len(s) > 1:
              tokens = s.split()
              for token in tokens:
                  if token in negation_words:
                      preposition_word += 1
                      
    print("Preposition word: ", preposition_word)#,": Percent: ", round((preposition_word/num_of_words) * 100,2))
    return preposition_word
       
def get_article_words(reviews):
   
    articles = ['a','an','the']
    
    num_articles = 0
        
    for i in range(len(reviews)):

       review = reviews[i]
       review = review.replace("�"," ")
       split_sentences  = re.split('[.!]', review) # re.split('[^a-zA-Z][]', review)
     

       for s in split_sentences:
           if len(s) > 1:
              tokens = s.split()
              for token in tokens:
                  if token in articles:
                      num_articles += 1
                      
    print("num_articles : ", num_articles) #,": Percent: ", round((num_articles/num_of_words) * 100,2))
    return num_articles
                     
      

    
#-------SR
#---- Probbaly Has some issues
#  do not use
def split_review_into_sentences_n_words(reviews):
    
   #split_sentences = review.split("�")
   
   num_of_sentences = 0
   #num_of_words = 0
   global  num_of_words;  
   
   word_length_reviews = []
   for i in range(len(reviews)):

       review = reviews[i]
       review = review.replace("�","")
       split_sentences  = re.split('[.!]', review) # re.split('[^a-zA-Z][]', review)
     
       sentences = []
       
       print("---Rev----: \n", review)
       #print("\n")
       word_length_review = 0
       for s in split_sentences:
          
           if len(s) > 1:
              #print("S: ", s)
              num_of_sentences += 1
              tokens = s.split()
              num_of_words += len(tokens)
              word_length_review += len(tokens)
              
              word_length_reviews.append(word_length_review)
              sentences.append(s)
              
    
   print("Sentence/review --> ", num_of_sentences/len(reviews), num_of_sentences, len(reviews))
   print("Words/review --> ", num_of_words/len(reviews), num_of_words, len(reviews))
   print("Words/sen --> ", num_of_words/num_of_sentences, num_of_words, num_of_sentences)
 

   #write_to_text_file(word_length_reviews, "/Users/russell/Downloads/length_yelp.txt")
   return sentences


def pos_tagger(reviews):
    import spacy
    from collections import Counter
    
    #import en_core_web_sm

    #nlp = en_core_web_sm.load()


    nlp = spacy.load("en_core_web_sm")
    
    sentences = []
    
    for i in range(len(reviews)):

        review = reviews[i]
        review = review.replace("�"," ")
        review = review.replace("\n"," ")
        split_sentences  = re.split('[.!]', review) # re.split('[^a-zA-Z][]', review)
     
        ##print(split_sentences[0])
        for sen in split_sentences:
            
            if len(sen) > 5 :
                #print("-> : ",sen)
                sentences.append(sen)
            
    ##return
    num_of_adj = 0
    num_of_verb = 0
    
    total_token = 0
    
    
    word_list_adj = []
    word_list_verb = []
    
    #print("Num o sentence: ", len(sentences))
    for i in range(len(sentences)):
     
        #if i == 100:
           # break
        #print(i)
        sentence = sentences[i]
        doc = nlp(sentence)
    
        for token in doc:
            total_token += 1
            ##print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
               # token.shape_, token.is_alpha, token.is_stop)
            #print( token.pos_)
            
            if token.pos_ == 'ADJ':
                num_of_adj += 1 
                word_list_adj.append(token.text)
                
            if token.pos_ == 'VERB':
                num_of_verb += 1 
                word_list_verb.append(token.text)
            
    print("total_token: ", total_token)
    print("num_of_adj: ", num_of_adj)
    print("num_of_verb: ", num_of_verb)  

    x = Counter(word_list_adj)
    x = x.most_common()[:50]
    print("ADJ \n\n",x) 
    
    x = Counter(word_list_verb)
    x = x.most_common()[:50]
    print("VERB \n\n",x)   
    
    print(": Percentage adj/verb: ", round((num_of_adj/num_of_words) * 100,2),round((num_of_verb/num_of_words) * 100,2))
    return num_of_adj, num_of_verb, total_token,len(sentences)




'''

All > 0 words
yelp
Score:  2314 --->  89.08540229885058
Length:  2258
AVerage: 76.72601702995703
STD: 10.520175436477343


Score:  2332 --->  95.93928571428573
Length:  2135
AVerage: 70.22201607475806
STD: 17.87572950670599
(base) Salims-MacBook-Pr





100 word-

Yelp: 
Score:  2313 --->  70.54785969935577
Length:  1169
AVerage: 76.6053361904909
STD: 8.429665153797838


Score:  2280 --->  71.5772650663943
Length:  64
AVerage: 73.22561393912102
STD: 8.125763807804164


50 word-



Score:  2314 --->  70.46166666666669
Length:  1776
AVerage: 76.67156266286774
STD: 9.003692904552445


Bangla
Score:  2319 --->  75.41692307692311
Length:  273
AVerage: 74.2023988391221
STD: 11.693763508911466


'''

#------------- Distinguisihing Language Nativenss -----
#----------- basedd on statistical attribbutes -------
#----------- Bangla (Bangladesh) and Yelp (English) Resutaurant -------
def get_features_both_dataset():
     
    
    data_1 = read_feature_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/Kaggle_BanglaRest.txt")
    data_2 = read_feature_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/Yelp.txt")
    
    ##data_1 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/BanglaFeature.txt")
   # data_2 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/YelpFeature.txt")
  
    
    #data_1 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/Kaggle_BanglaRest.txt")
   # data_2 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/Yelp.txt")
    
    
    data = data_1 + data_2
    
    label = [0] * len(data_1) + [1] *  len(data_2)
    
    print("data ", len(data))
    print("label ", len(label))
    
    #return
    classify_language_nativeness(data,label)
    

def classify_language_nativeness(data, label):
     
    import random
       
    for itr in range (1):  
  
        c = list(zip(data, label))
        random.shuffle(c)
        data, label  = zip(*c)
        
        print ("~~~~~~~ ",data[0],label[0] )
        print("Total Data all directory: ",len(label), type(label), label[1])
        
        
        data = np.array(data)
        label = np.asarray(label)
        #label=label.astype('int')
           
    
        total_f1 = 0
        total_acc = 0
        total_precison = 0
        total_recall = 0
        
    
        num_of_fold = 10 #10 # 3
        #kf = KFold(n_splits=num_of_fold)
        kf = StratifiedKFold(n_splits=num_of_fold)
  
        
        for train_index, test_index in kf.split(data, label):
    
            #print("train_index: ", train_index)
            classifier = Logistic_Regression_Classifier() 
            
            #classifier = SVM_Classifier() 
            #classifier = SGD_Classifier() 
            
            #classifier =  ExtraTree_Classifier()
            
            
            #classifier =  RandomForest_Classifier()
            
            label_train, label_test = label[train_index], label[test_index]
            data_train, data_test = data[train_index], data[test_index]
            
            from imblearn.over_sampling import SMOTE

            #sampler = SMOTE()
            #data_train, label_train = sampler.fit_sample(data_train, label_train)
           
            prediction = classifier.predict(data_train, label_train, data_test)
        
            performance = Performance() 
        
            
            
            #conf_matrix,precision,  recall, f1_score, acc = performance.get_results(label_test, prediction)
        
        
            conf_matrix,precision,  recall, f1_score, acc = performance.get_results(label_test, prediction)
    
            #print("F1# ",f1_score)
            total_f1 += f1_score
            total_acc += acc
            total_precison  += precision
            total_recall += recall
            
            print(conf_matrix)
            #print
            
            #print
        
        print("Overall")
        #print("\n\nEnglish P-R-F1-Acc: ",round(total_english_precison/num_of_fold,3), round(total_english_recall/num_of_fold,3)   , round(total_english_f1/num_of_fold, 3), round(total_english_acc/num_of_fold,3))
        print("Results: P:R:F1-Acc", round(total_precison/num_of_fold,3), round(total_recall/num_of_fold,3) , round(total_f1/num_of_fold,3), round(total_acc/num_of_fold, 3))
        print("\n\n------")
    
def get_tfidf(X):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(X)
    #print(X)
    return X


def get_class_distribution(label):
    
    dic = {}
    label = label.astype('int')
    for value in label:
        if value in dic:
            dic[value] =  dic[value] + 1
        else:
            dic[value] =  1
            
    for key in dic:
        print(key, dic[key])
    
def get_bangla_english_reviews():
    data_11, label_11 = getBanglaRestaurantReviews()
    data_12, label_12 = getKaggleRestaurantReviews()
    
    
    print("Length of BanglaRestaurant  ", len(data_11))
     
    print("Length of KaggleRestaurant Bangladesh ", len(data_12))
    print("Total: ", len(data_11) + len(data_12))
    
    data_1 = np.concatenate((data_11, data_12), axis=0)
    label_1 = np.concatenate((label_11, label_12), axis=0)
    
    get_class_distribution(label_1)
    
    # Write to ffile
    #write_to_CSV("/Users/russell/Documents/NLP/Restaurant/Data/Combined_Kaggle_BanglaRest_4987.csv", data_1, label_1)
    
    #return
    print("\n\n\n\n")
   # data_1, _ = getBanglaRestaurantReviews()
    data_2, label_2 = get_data_n_label_from_yelp_review_X_pos_Y_neg("/Users/russell/Downloads/yelp.xlsx", "yelp",1676, 3311)
   
    #data_2, label_2 = get_data_n_label_from_yelp_review_X_pos_Y_neg("/Users/russell/Downloads/yelp.xlsx", "yelp",2579, 3500)
   
    print("Length of data-2 ", len(data_2))
    #data_1 = data_1[:]
    data_2 = data_2[:len(data_1)] ##data_2[:5000] 
    get_class_distribution(label_2)
    
    #write_to_CSV("/Users/russell/Documents/NLP/Restaurant/Data/Yelp_4987.csv", data_2, label_2)
   
    data = np.concatenate((data_1, data_2), axis=0)
    #data = data_1 + data_2
    
    label = [0] * len(data_1) + [1] *  len(data_2)
    

    return data, label


def get_class_relevant_words():
    import pandas as pd
    import random
    
    data, label = get_ELP_data_four_classes()
    print("Data length",len(data))
    
    c = list(zip(data, label))
    random.shuffle(c)
    data, label  = zip(*c)
    
    from sklearn.linear_model import LogisticRegression
    #from sklearn.linear_model import SVMClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer


    label = np.asarray(label)
      
    data = np.asarray(data)
        

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X_train = vectorizer.fit_transform(data)

    clf =  LogisticRegression() #SVM_Classifier() # 
    clf.fit(X_train, label)
    
   # print(clf.coef_[0])
   # print(vectorizer.get_feature_names())

    for i in range(4):
        important_tokens = pd.DataFrame(
            data=clf.coef_[i],
            index=vectorizer.get_feature_names(),
            columns=['coefficient']
        ).sort_values(by = 'coefficient',ascending=False)
        
        n_top_words = 50
        print("\n###\n")
        print(important_tokens.head(n_top_words).index.values)
       
        #print(important_tokens.head(30).index.values,important_tokens.head(30).values)
         
    
# Get data for all the four classes and then apply ML classifiers
def apply_ML_TFIDF_LN():
    
    
    ##return 
    
    import random 
    
    
    #data, label = get_bangla_english_reviews()
    
    #data, label = get_ELP_data()
    
    data, label = get_ELP_data_four_classes()
    
    # Uncomment for reading stylistic data
    #data, label = get_extracted_features_ELP_four_classes()
    
    
    print("Data length",len(data))
    
    c = list(zip(data, label))
    random.shuffle(c)
    data, label  = zip(*c)
   # print("Total Data all directory: ",len(label), type(label), label[1])
    
    
    
    label = np.asarray(label)
    
    data = np.asarray(data)
    
    #Uncomment when text is read
    data = get_tfidf(data)
    
    num_of_fold = 5
    from sklearn.model_selection import StratifiedKFold 
    kf = StratifiedKFold(n_splits=num_of_fold)
    
    #X_train_english, Y_train_english  = get_sampled_data(X_train_english, Y_train_english)
    
    
    
    total_f1 = 0
    total_acc = 0
    total_precison = 0
    total_recall = 0

    
    #prediction_english_all = []
    #prediction_bengali_all = []

    
    conf_matrix_all = [[0] * 4] * 4

    for train_index, test_index in kf.split(data, label):

        classifier = Logistic_Regression_Classifier() 
        
        #classifier = Ridge_Regression_Classifier()
        
        
        #classifier = RandomForest_Classifier()
        
        classifier = SVM_Classifier()
        #classifier = KNN_Classifier()
        
        #classifier = SGD_Classifier()
        
        
        #classifier = LDA_Classifier()
        
        #classifier = MultinomialNB_Classifier()
        
        
        label_train, label_test = label[train_index], label[test_index]
        data_train, data_test = data[train_index], data[test_index]
       
        prediction = classifier.predict(data_train, label_train, data_test)
    
        performance = Performance() 
    
        
        
        conf_matrix,precision,  recall, f1_score, acc = performance.get_results(label_test, prediction)
    
        #print("Bengali F1#",f1_score)
        total_f1 += f1_score
        total_acc += acc
        total_precison  += precision
        total_recall += recall
        
        conf_matrix_all += conf_matrix
        print(conf_matrix)
        #print
    
    
    
    
    print("\n\n\n",conf_matrix_all)
    print("Overall")
    #print("\n\nEnglish P-R-F1-Acc: ",round(total_english_precison/num_of_fold,3), round(total_english_recall/num_of_fold,3)   , round(total_english_f1/num_of_fold, 3), round(total_english_acc/num_of_fold,3))
    print("\Total: P:R:F1-Acc", round(total_precison/num_of_fold,3), round(total_recall/num_of_fold,3) , round(total_f1/num_of_fold,4), round(total_acc/num_of_fold, 4))
    #print("\n\n------")

def get_data_named_entity_removed(reviews): 
    import spacy
    nlp = spacy.load('en_core_web_sm')
    ne_removed_documents = []
    
    #for i in range (30): #(len(reviews)):
    for i in range (len(reviews)): 
        review = reviews[i]
        review = review.replace("#", "")
        review = review.replace(".", " ")
        #print(i, "\n" , review)
        doc = nlp(review)
        
        
        #print("\nEntities:", doc.ents)
        if doc.ents:
            for ent in doc.ents:
                
                if ent.label_  != 'GPE' and  ent.label_  != 'PERSON' and ent.label_  != 'ORG':
                    continue
                #if ent.label_  != 'PERSON':
                   # continue
                
                print("Label: ", ent.text, ent.label_)
                #print()
                review= review.replace(ent.text, "")
                #print("After Entity removed", doc)
                #start = ent.start_char
                #end = ent.end_char
        ne_removed_documents.append(review)
                
    return ne_removed_documents
                

#def get_named_entity(data):
    #import spacy
    #nlp = spacy.load('en core web sm')
    
      
def read_single_features():
    data_1 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/Kaggle_BanglaRest.txt")
    data_2 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/Yelp.txt")
  
    
    num_1 = [samples[7] for samples in  data_1]
    num_2 = [samples[7] for samples in  data_2]
    
    #print(num_1[0])
    
    write_to_text_file(num_1, "/Users/russell/Downloads/num_bangla_7.txt")
    write_to_text_file(num_2, "/Users/russell/Downloads/num_eglish_7.txt")
    

#---------- Extract Features  ----
#---------- Bangla and Yelp  ----
def extract_features():
    
    #çX_data, Y_label = getBanglaRestaurantReviews()
    #X_data, Y_label = get_data_n_label_from_yelp_review_1702_pos_613_neg("/Users/russell/Downloads/yelp.xlsx", "yelp")
    
    
    
    #X_data, Y_label = read_data_from_excel("/Users/russell/Documents/NLP/Restaurant/Data/Combined_Kaggle_BanglaRest_4987.xlsx","Combined_Kaggle_BanglaRest_4987")
   # X_data, Y_label = read_data_from_excel("/Users/russell/Documents/NLP/Restaurant/Data/Combined_Kaggle_BanglaRest_4987.xlsx","Combined_Kaggle_BanglaRest_4987")
    #write_to_CSV(X_data, Y_label)
    X_data, Y_label = read_data_from_excel("/Users/russell/Documents/NLP/Restaurant/Data/Final/Yelp_4987.xlsx","Yelp_4987")
  
    
    #return 
    
    sentiment_dictionary_BingLiu()
    get_vader_lexicon()

    num_of_review = len(X_data)
    
    features = []
    for i in range (num_of_review):
        temp = []
        #print("\n", X_data[i])
        temp.append(X_data[i].lower())
        
        num_of_neg = get_negation_words(temp)
        num_of_prepo =  get_preposition (temp)
        num_of_article =  get_article_words(temp)
        
        num_of_Bing = round (getLexiconCoverageOfBingLiu(temp),3)
        num_of_VADER = round(getLexiconCoverageOfVADER(temp),3)
        
        num_of_adj, num_of_verb, num_of_words, num_of_sentence = pos_tagger(temp)
        
        if num_of_sentence == 0:
            num_of_sentence = 1
        num_of_words_sentence = round (num_of_words / num_of_sentence,3)
        
        in_clause, out_clause = compute_dependent_clause_ratio(temp)
        
        clause_ratio = round(in_clause/ (in_clause + out_clause),3)
        
        print("\nReview : ", i)
        #print(num_of_words,num_of_sentence,num_of_words_sentence,num_of_neg,num_of_prepo,num_of_article,num_of_adj,num_of_verb, clause_ratio)
       # print(num_of_Bing,num_of_VADER)
        
        feature =  str(num_of_words) + "," + str(num_of_sentence) + ","  + str(num_of_words_sentence)  + \
        "," + str(num_of_neg) + "," + str(num_of_prepo) + "," + str(num_of_article) +"," + str(num_of_adj) + \
        "," + str(num_of_verb) + "," + ","  + str(clause_ratio) + ","+ str(num_of_Bing ) \
        + "," + str(num_of_VADER)
        
        features.append(feature)

    #write_to_text_file(features, "/Users/russell/Downloads/Kaggle_BanglaRest.txt")  
    write_to_text_file(features, "/Users/russell/Downloads/Yelp.txt")      
    
def extract_features_for_ELP_four_classes():
   
    #X_data, Y_label = read_data_from_excel("/Users/russell/Documents/NLP/Restaurant/Data/Final/Yelp_4987.xlsx","Yelp_4987")
  
    #X_data = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Finland.txt")
    X_data = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Kenya.txt")
    X_data = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/China.txt")
    X_data = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/Bangladesh.txt")
   # 
    
    #return 
    
    sentiment_dictionary_BingLiu()
    get_vader_lexicon()

    num_of_review = len(X_data)
    
    features = []
    for i in range (num_of_review):
        temp = []
        #print("\n", X_data[i])
       
        
        review = X_data[i].lower()
        print("\nReview : ", i, "\n", review)
   
        temp.append(review)
        
        
        num_of_words, num_of_sentence = get_num_of_words_n_sentences(review)
        
        num_of_neg = get_negation_words(temp)
        
        
        num_of_prepo =  get_preposition (temp)
        num_of_article =  get_article_words(temp)
        
        num_of_Bing = round (getLexiconCoverageOfBingLiu(temp),3)
        num_of_VADER = round(getLexiconCoverageOfVADER(temp),3)
        
        num_of_adj, num_of_verb, num_of_words, num_of_sentence = pos_tagger(temp)
        
        if num_of_sentence == 0:
            num_of_sentence = 1
        num_of_words_sentence = round (num_of_words / num_of_sentence,3)
        
        in_clause, out_clause = compute_dependent_clause_ratio(temp)
        
        clause_ratio = round(in_clause/ (in_clause + out_clause),3)
        
        num_of_neg = round(num_of_neg/ num_of_words,5)
        num_of_prepo = round(num_of_prepo/num_of_words,5)
        num_of_article = round(num_of_article/ num_of_words,5)
        num_of_adj = round(num_of_adj/ num_of_words,5)
        num_of_verb = round(num_of_verb /num_of_words,5)
        
      
        #print(num_of_words,num_of_sentence,num_of_words_sentence,num_of_neg,num_of_prepo,num_of_article,num_of_adj,num_of_verb, clause_ratio)
       # print(num_of_Bing,num_of_VADER)
        
        feature =  str(num_of_words) + "," + str(num_of_sentence) + ","  + str(num_of_words_sentence)  + \
        "," + str(num_of_neg) + "," + str(num_of_prepo) + "," + str(num_of_article) +"," + str(num_of_adj) + \
        "," + str(num_of_verb) + ","  + str(clause_ratio) + ","+ str(num_of_Bing ) \
        + "," + str(num_of_VADER)
        
        print(feature)
        features.append(feature)

    #write_to_text_file(features, "/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/features/features_Finland.txt")    
    #write_to_text_file(features, "/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/features/features_Kenya.txt")    
    #write_to_text_file(features, "/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/features/features_China.txt")    
    write_to_text_file(features, "/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/features/features_Bangladesh.txt")      
      
    
def get_extracted_features_ELP_four_classes():
     
    
    
    ##data_1 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/BanglaFeature.txt")
   # data_2 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/YelpFeature.txt")
  
    
    #data_1 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/Kaggle_BanglaRest.txt")
   # data_2 = read_data_from_textfile("/Users/russell/Documents/NLP/Restaurant/Data/Final/features/Yelp.txt")
    
   
    
   data_1 = read_feature_data_from_textfile("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/features/features_Finland.txt")
   data_2 = read_feature_data_from_textfile("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/features/features_Kenya.txt")
   data_3 = read_feature_data_from_textfile("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/features/features_China.txt")
   data_4 = read_feature_data_from_textfile("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/features/features_Bangladesh.txt")
   
   
    
   data = data_1 + data_2 + data_3 + data_4 
    
   label = [0] * len(data_1) + [1] *  len(data_2) + [2] * len(data_3) + [3] *  len(data_4)
    
   print("data ", len(data))
   print("label ", len(label))

   return np.array(data), np.array(label)
   #return data, label
    #return
    #classify_language_nativeness(data,label)



def main():
    
    
    #extract_features_for_ELP_four_classes()
    #return
    apply_ML_TFIDF_LN()
    return
    
    
    get_class_relevant_words()
    return
    
    #review = ["Of course we had to see what the buzz was all about, so I booked this restaurant to treat a foodie friend of mine. Hard to say which one is the original, is it Bas Bas Bistro or Bas Bas Kulma. I booked a table to Kulma (which is not actually in the street corner as the name suggests). Tables can be booked only a month in advance and be quick about it!Despite the covid restrictions I don’t think this restaurant could fit any more people in this restaurant… Then the food: good food, but the menu idea needs tweaking since my friend got her sea bass and pork portion at the same time. You really cannot eat two mains at the same time if they’re fish and meat. For me this was not a problem since I ordered pork and carrots for my main dishes. Also there could be wine recommendations for each dish to make life easier, especially since our waitress told us this was her first shift. Also had to remind the staff that we’d like our wine with the food and then again the glass of port ordered with the dessert.I thought that the food was too salty, especially my main dishes, but my friend thought that it was a bit salty but not too much. To be honest, I didn’t get why this restaurant is so popular."]
    
    #review = ["Had definitely one of my best dinner. Food was done with very fresh and tasty ingredients, served quite straight forward way, but still stylish. Menu is not large, but you find your chose. Also, my wife has gluten free and sea food allergy, and even though we did not remember to tell it in advance, everything was taken into account in very professional way. I do not like raw wines, and we had very nice traditional and tasty wines over the dinner. Just go! But book!."]
    
    #review = ["The menu has 6 choices for a starter, two for a mid course and two for a main. Their recommendation is to choose 2 per starter, 1 per mid and 1 per main, but you can of course vary from this if you want - we found this a good guideline. The menu changes every month or so with the seasons and is not available to view online as far as I understand. So you have to go to know what is on offer which is kind of nice in this day and age. "]
    
    #Flesch_Kincaid_Grade_Level(review)
    #return
    #evaluate_indepenent_set()
    #return

    #read_single_features()
    #return
    
  
    #extract_features()
    #return
    
    #get_features_both_dataset()
    #return
   
    #get_vader_lexicon()
    #sentiment_dictionary_BingLiu()
    
    #X_data, Y_label = read_data_from_excel("/Users/sazzed/Documents/Other/Research/Restaurant/Data/Final/Combined_Kaggle_BanglaRest_4987.xlsx","Combined_Kaggle_BanglaRest_4987")
    #X_data, Y_label = read_data_from_excel("/Users/sazzed/Documents/Research/Restaurant/Data/Final/Yelp_4987.xlsx","Yelp_4987")
    
    country = "Bangladesh" # "China"#  "kenya"# "Finland" # "kenya"# "China"#  "Bangladesh" # "kenya" #"Finland" #"kenya" #"China"#" "Bangladesh" #"mayanmar" #"Bangladesh" #"China"#"kenya" # "Finland" #  "Bangladesh"#"China"#" #
    X_data = read_text_data_independent_set("/Users/sazzed/Documents/Other/Research/Restaurant/ELP-Data/" + country + ".txt")
    
    pos_tagger(X_data)
    #getLexiconCoverageOfBingLiu(X_data)  
    #findLexiconCoverage(X_data)
    
    #extract_word_from_reviews(X_data)
    #compute_dependent_clause_ratio(X_data)
    return
    
    #Flesch_Kincaid_Grade_Level(X_data)
    return
   # print(len(X_data), X_data[10])
    #Flesch_reading_ease(X_data)
    #return
    #X_data, Y_label = getKaggleRestaurantReviews()
    ##X_data, Y_label = getBanglaRestaurantReviews()
    ##X_data, Y_label = get_data_n_label_from_yelp_review_1702_pos_613_neg("/Users/russell/Downloads/yelp.xlsx", "yelp")
    
    #write_to_CSV(X_data, Y_label)
    #return;
    
    
    
    X_data = X_data[:71]
    
    # Text Statistics
    split_review_into_sentences_n_words(X_data)
    get_num_of_unique_words(X_data)
    #return
    get_negation_words(X_data)
    get_article_words(X_data)
    pos_tagger(X_data)
    get_preposition(X_data)
    #findLexiconCoverage(X_data)    
    
    #compute_dependent_clause_ratio(X_data)
    
    #get_common_words()
    
    #pos_tagger(X_data)
    
    #get_num_of_unique_words(X_data)
   
    return
   
    
    
    ##return

    #get_num_of_words_n_sentences(np.array(X_data))
    
  
    #return 
    
    #split_review_into_sentences_n_words(X_data)
    return
        
    
    
    
    #split_review_into_sentences_n_words(X_data[:2315])
    #findLexiconCoverage(X_data)
    
    return
    print("Yelp")
    #X_data, Y_label = get_data_n_label_from_yelp_review_1702_pos_613_neg("/Users/russell/Downloads/yelp.xlsx", "yelp")
    #split_review_into_sentences_n_words(X_data[:])
    
    return
    #get_data_n_label_from_yelp_review("/Users/russell/Downloads/yelp.xlsx", "yelp")
    
    extract_word_from_reviews(X_data)
    ##compute_dependent_clause_ratio(X_data)

if __name__ == '__main__':
    #parse_json_data()
    main()


