import math
from collections import Counter

from flask import Flask, jsonify
import numpy as np
from scipy.optimize import minimize
import spacy
import neuralcoref
import nltk

app = Flask(__name__)


@app.route('/allMetrics', methods=['POST'])
def allMetrics(texts):
    zipf = getZipf(texts)
    corref = getCorreferences(texts)
    wordLength = getWordLength(texts)
    word2vec = getWord2VecContinuity(texts)
    conditionalEntropy = getConditionalEntropy(texts)
    conditionalEntropyPOS = getConditionalEntropyPOS(texts)
    numSynSets = getNumSynSets(texts)
    posDistribution = getPOSDistribution(texts)
    return jsonify()


def getZipf(texts):
    zipfVals = []
    for text in texts:
        c = Counter(text)
        currentArr = []
        for val in c:
            currentArr.append(c[val])
            x = np.arange(1, len(currentArr) + 1, 1)
            s = np.array(currentArr).astype(int)
            s[::-1].sort()
            s_best = minimize(loglik(x=x), [2])
            zipfVals.append(s_best.x[0])
    return zipfVals


def getCorreferences(texts):
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    correfs = []
    for text in texts:
        corref = {}
        doc = nlp(text)
        for item in doc._.coref_clusters:
            size = len(item)
            if size in corref:
                corref[size] += 1
            else:
                corref[size] = 1
        correfs.append(corref)
    return correfs


def getWordLength(texts):
    wordLength=[]
    for text in texts:
        wordLen={}
        for word in text.split():
            length=len(word)
            if length in wordLen:
                wordLen[length]+= 1
            else:
                wordLen[length]=1
        wordLength.append(wordLen)
    return wordLength


def getConditionalEntropy(texts):
    conditionalEntropy=[]
    for text in texts:
        conditionalEntropy.append(computeCompleteHillbergConditional(text, 8))


def getConditionalEntropyPOS(texts):
    conditionalEntropy=[]
    for text in texts:
        POStext=turnToPOS(text)
        conditionalEntropy.append(computeCompleteHillbergConditional(POStext, 8))

def turnToPOS(text):
    posText=""
    for sentence in text.split("\n"):
        tokens = nltk.word_tokenize(sentence)
        postag = nltk.pos_tag(tokens)
        for tag in postag:
            posText+=" "+str(tag[1])
    return posText


def getWord2VecContinuity(texts):
    return None


def getNumSynSets(texts):
    nlp = spacy.load("en_core_web_sm")
    synsets = []
    for text in texts:
        synset = {}
        doc = nlp(text)
        for token in doc:
            distance = abs(token.idx - token.head.idx)
            if distance in synset:
                synset[distance] += 1
            else:
                synset[distance] = 1
        synsets.append(synset)
    return synsets


def getPOSDistribution(texts):
    distributions = []
    for text in texts:
        tagAux = {}
        for sentence in text.split("\n"):
            tokens = nltk.word_tokenize(sentence)
            postag = nltk.pos_tag(tokens)
            for tag in postag:
                if tag[1] in tagAux:
                    tagAux[tag[1]] += 1
                else:
                    tagAux[tag[1]] = 1
        sumVal = sum(tagAux.values())
        if sumVal == 0:
            sumVal = 1
        factor = 1.0 / sumVal
        for k in tagAux:
            tagAux[k] = tagAux[k] * factor
        distributions.append(tagAux)
    return distributions


def loglik(x, b):
    # Power law function
    Probabilities = x ** (-b)

    # Normalized
    Probabilities = Probabilities / Probabilities.sum()

    # Log Likelihoood
    Lvector = np.log(Probabilities)

    # Multiply the vector by frequencies
    Lvector = np.log(Probabilities) * s

    # LL is the sum
    L = Lvector.sum()

    # We want to maximize LogLikelihood or minimize (-1)*LogLikelihood
    return (-L)


def computeWindowsConditional(text,windowSize):
    frequenciesNgram={}
    conditionalFrequencies={}

    words=text.split()
    for j in range(0,len(words)-windowSize):
        ngram=''
        first=True
        for t in range(j,j+windowSize):
            word=words[t]
            if first:
                ngram=word
                first=False
            else:
                    ngram=ngram+"_"+word
        if not ngram in frequenciesNgram:
                frequenciesNgram[ngram]=1
        else:
                frequenciesNgram[ngram]=frequenciesNgram[ngram]+1
    for j in range(0,len(words)-windowSize):
        ngram=''
        first=True
        for t in range(j,j+windowSize):
            word=words[t]
            if first:
                ngram=word
                first=False
            else:
                ngram=ngram+"_"+word
        if not ngram in conditionalFrequencies:
            conditionalFrequencies[ngram]={}
            conditionalFrequencies[ngram][words[j+windowSize]]=1
        else:
            if words[j+windowSize] in conditionalFrequencies[ngram]:
                conditionalFrequencies[ngram][words[j+windowSize]]=conditionalFrequencies[ngram][words[j+windowSize]]+1
            else:
                conditionalFrequencies[ngram][words[j+windowSize]]=1
    return frequenciesNgram,conditionalFrequencies


def computeHillbergLawConditional(windowsy,windowsxy):
    n=sum(windowsy.values())
    tot=n
    sumEntropy=0
    if len(windowsy)>=1 and tot>=1:
        for l in windowsy.keys():
            px=windowsy[l]/tot
            if l in windowsxy:
                windowxy=windowsxy[l]
                for key in windowxy.keys():
                    pxy=windowxy[key]/tot
                    logp=-math.log2(pxy/px)
                    logp=logp*pxy
                    sumEntropy=sumEntropy+logp
    return sumEntropy

def computeCompleteHillbergConditional(text,windowSize):
    hillbergVals=[]
    for i in range(1,windowSize):
        windowsy,windowsxy=computeWindowsConditional(text,i)
        hillbergVals.append(computeHillbergLawConditional(windowsy,windowsxy,i))
    return hillbergVals

if __name__ == '__main__':
    app.run()
