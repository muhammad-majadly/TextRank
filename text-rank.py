# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:16:22 2021

@author: mmaja
"""

#from functools import partial
import numpy as np
import nltk
nltk.download('punkt')
from w2vec2 import Word2Vec
import numpy 
import codecs
from scipy.spatial import distance
from stanfordTagger2 import posTager2 
#from pagerank import pagerank


# PageRank Algorithm 
def pagerank(matrix, eps=0.0001, d=0.85):
    n = len(matrix)
    probs = np.ones(n)/n
    for i in range(4):
        new_p = np.ones(n) * (1-d)/n + d*matrix.T.dot(probs)
        delta = abs(new_p-probs).sum()
        if delta <= eps:
            break
        probs = new_p
    return new_p

def magnitude(vec):
    return np.linalg.norm(vec)

# similarity calculating
def calculate_similarity(tokens1, tokens2): 
    
    return distance.cosine(tokens1,tokens2)

# return similarity matrix
def build_sim_matrix(sentences_vector):
    n = len(sentences_vector)
    matrix = np.ones((n, n))
    for i, sent1 in enumerate(sentences_vector):
        for j, sent2 in enumerate(sentences_vector):
            sim = calculate_similarity(sent1, sent2)
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


# return w2vec for a sentence using the nouns
def getW2vecSentence(sentence,pos):
    res=numpy.zeros(300)
    tokens = nltk.word_tokenize(sentence)
    nounsCount=0
    for i in range(len(tokens)):
        if isNoun(tokens[i],pos):
            nounsCount+=1
            if tokens[i] in w2vecModel.wv.vocab:
                res+= numpy.array(list(w2vecModel.wv[tokens[i]]))

    res/=nounsCount #normalization

    return res  

# return w2vec for a sentence using all the words
def getW2vecSentence2(sentence):
    res=numpy.zeros(300)
    tokens = nltk.word_tokenize(sentence)
    for i in range(len(tokens)):
            if tokens[i] in w2vecModel.wv.vocab:
                res+= numpy.array(list(w2vecModel.wv[tokens[i]]))

    res/=len(tokens) #normalization

    return res 

# return if w is a noun
def isNoun(w,pos):
    return pos[w]=='NN' or pos[w]=='NP' or pos[w]=='NNP' or pos[w]=='NNS'

# preprocess for POS tags        
def buildPOS(allText):
    allTokens = nltk.word_tokenize(allText)
    allPos = posTager2(allTokens)
    posHash={}
    for i in range(len(allTokens)):
        posHash[allTokens[i]]=allPos[i]
    return  posHash    


## main()
    
## reading the article input, the articles splited by # 
f= open("input.txt","r",encoding='utf-8-sig')    
textFiles=f.read()
textFiles=textFiles.split("#")
                          
# open the output file                          
file1 = codecs.open("out1.txt","w","utf-8") 

# the Algrothim
for k in range(len(textFiles)):
    
    text= textFiles[k]    
    sentences = nltk.tokenize.sent_tokenize(text) # split the text sentences

    w2vecModel =  Word2Vec.load('ar_wiki_word2vec')
    pos=buildPOS(text) #get POS tags 
    w2vecSent=[]
    w2vecSent = [0 for i in range(len(sentences))] #initilaize sentence ranks
    for i in range(len(sentences)): # get w2vec for the sentences
        w2vecSent[i] = getW2vecSentence(sentences[i],pos)
 
    matrix = build_sim_matrix(w2vecSent)
    sentRank = pagerank(matrix)
    
    #sort the sentence by the rank
    sentencesRes = [x for _,x in sorted(zip(sentRank,sentences))]
    
    #sort the rankSentence for print
    sentRank.sort()
    
    #print the output into the file 
    file1.write("\n------------------------- article number : " + str(k) + "-------------------------\n")
    for i in range(len(sentencesRes)-1,-1,-1):
        file1.write (sentencesRes[i] + " - " + str(sentRank[i]) +'\n')
    
      
file1.close() 