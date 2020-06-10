import math
from collections import Counter

import neuralcoref
import nltk
import spacy
from flask import Flask, request
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='Text Metrics API',
          description='A text metric extractor',
          )

ns = api.namespace('Metrics', description='Text metrics')

textsModel = api.model('Text',
                      {
    'id': fields.Integer(readOnly=True, description='The text identifier'),
    'text': fields.String(required=True, description='The text to analyze')
})



@ns.route('/allMetrics')
class Metrics(Resource):

    @ns.doc('Obtain metrics')
    #@ns.marshal_with(metricsModel,as_list=True)
    @ns.expect([textsModel])
    @ns.doc(params={'payload': 'Expected array of texts, defined by their ID, returns a map with each quality '})
    def post(self):
        toRet = []
        data = request.json
        for text in data:
            corref = self.getCorreferences(text["text"])
            numSynSets = self.getNumSynSets(text["text"])
            posDistribution = self.getPOSDistribution(text["text"])
            auxText=text["text"].replace(',', '' '').replace('.', ' ').replace(':', " ").replace('-', ' ')
            zipf = self.getZipf(auxText)
            wordLength = self.getWordLength(auxText)
            conditionalEntropy = self.getConditionalEntropy(auxText)
            conditionalEntropyPOS = self.getConditionalEntropyPOS(auxText)
            conditionalEntropyChar = self.getConditionalEntropyChar(auxText)
            metricObject = {"zipf": zipf,
                            "corref": corref,
                            "wordLength": wordLength,
                            "conditionalEntropy": conditionalEntropy,
                            "conditionalEntropyPOS": conditionalEntropyPOS,
                            "numSynSets": numSynSets,
                            "posDistribution": posDistribution,
                            "conditionalEntropyChar": conditionalEntropyChar,
                            }
            returnObject= {
                "metrics":metricObject,
                "id": text["id"]

            }
            toRet.append(returnObject)
        return toRet

    x = None
    s = None

    def getZipf(self, text):
        c = Counter(text.split())
        currentArr = []
        for val in c:
            currentArr.append(c[val])
        currentArr.sort(reverse=True)
        return currentArr

    def getCorreferences(self, text):
        nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(nlp)
        corref = {}
        doc = nlp(text)
        for item in doc._.coref_clusters:
            size = len(item)
            if size in corref:
                corref[size] += 1
            else:
                corref[size] = 1
        toNormalizeBy=sum(corref.values())
        if toNormalizeBy==0:
            factor=1
        else:
            factor = 1.0 / toNormalizeBy
        for item in corref:
            corref[item]=corref[item]*factor
        return corref

    def getWordLength(self, text):
        wordLen = {}
        for word in text.split():
            length = len(word)
            if length in wordLen:
                wordLen[length] += 1
            else:
                wordLen[length] = 1
        toNormalizeBy=sum(wordLen.values())
        if toNormalizeBy==0:
            factor=1
        else:
            factor = 1.0 / toNormalizeBy
        for item in wordLen:
            wordLen[item]=wordLen[item]*factor
        return wordLen

    def getConditionalEntropyChar(self, text):
        return self.computeCompleteHillbergConditional(list(text), 14,True)

    def getConditionalEntropy(self, text):
        return self.computeCompleteHillbergConditional(text, 12)

    def getConditionalEntropyPOS(self,text):
        POStext = self.turnToPOS( text)
        return self.computeCompleteHillbergConditional(POStext, 8)

    def turnToPOS(self, text):
        posText = ""
        for sentence in text.split("\n"):
            tokens = nltk.word_tokenize(sentence)
            postag = nltk.pos_tag(tokens)
            for tag in postag:
                posText += " " + str(tag[1])
        return posText

    def getNumSynSets(self, text):
        nlp = spacy.load("en_core_web_sm")
        synset = {}
        doc = nlp(text)
        for token in doc:
            distance = abs(token.idx - token.head.idx)
            if distance in synset:
                synset[distance] += 1
            else:
                synset[distance] = 1
        toNormalizeBy=sum(synset.values())
        if toNormalizeBy==0:
            factor=1
        else:
            factor = 1.0 / toNormalizeBy
        for item in synset:
            synset[item]=synset[item]*factor
        return synset

    def getPOSDistribution(self, text):
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
        return tagAux


    def computeWindowsConditional(self, text, windowSize):
        frequenciesNgram = {}
        conditionalFrequencies = {}
        if not list:
            words = text.split()
        else:
            words=text
        for j in range(0, len(words) - windowSize):
            ngram = ''
            first = True
            for t in range(j, j + windowSize):
                word = words[t]
                if first:
                    ngram = word
                    first = False
                else:
                    ngram = ngram + "_" + word
            if not ngram in frequenciesNgram:
                frequenciesNgram[ngram] = 1
            else:
                frequenciesNgram[ngram] = frequenciesNgram[ngram] + 1
            if not ngram in conditionalFrequencies:
                conditionalFrequencies[ngram] = {}
                conditionalFrequencies[ngram][words[j + windowSize]] = 1
            else:
                if words[j + windowSize] in conditionalFrequencies[ngram]:
                    conditionalFrequencies[ngram][words[j + windowSize]] = conditionalFrequencies[ngram][
                                                                               words[j + windowSize]] + 1
                else:
                    conditionalFrequencies[ngram][words[j + windowSize]] = 1

        #for j in range(0, len(words) - windowSize):
            # ngram = ''
            #first = True
            #for t in range(j, j + windowSize):
                #   word = words[t]
                #if first:
                    #    ngram = word
                    #first = False
                    #else:
        # ngram = ngram + "_" + word
        return frequenciesNgram, conditionalFrequencies

    def computeHillbergLawConditional(self, windowsy, windowsxy, list=False):
        n = sum(windowsy.values())
        tot = n
        sumEntropy = 0
        if len(windowsy) >= 1 and tot >= 1:
            for l in windowsy.keys():
                px = windowsy[l] / tot
                if l in windowsxy:
                    windowxy = windowsxy[l]
                    for key in windowxy.keys():
                        pxy = windowxy[key] / tot
                        logp = -math.log2(pxy / px)
                        logp = logp * pxy
                        sumEntropy = sumEntropy + logp
        return sumEntropy

    def computeCompleteHillbergConditional(self, text, windowSize, list=False):
        hillbergVals = []
        for i in range(1, windowSize):
            windowsy, windowsxy = self.computeWindowsConditional(text, i)
            hillbergVals.append(self.computeHillbergLawConditional(windowsy, windowsxy,list))
        return hillbergVals


if __name__ == '__main__':
    app.run()
