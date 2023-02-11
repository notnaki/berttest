import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import pickle

stemmer = LancasterStemmer()
intents = json.load(open('train-v2.0.json'))
intents = intents['data']
words,labels = [], []
docx, docy = [], []

discard = [
    '?',
    '!',
    '.',
    ','
]

"""
title : tag
paragraphs->qas->question : question
"""
indx = 0
for intent in intents:

    if indx < 101:
        for questions in intent["paragraphs"]:
            questions = questions['qas']
            for question in questions:
                wrds = nltk.word_tokenize(question['question'])
                words.extend(wrds)
                docx.append(wrds)
                docy.append(str(indx)+question['id'])
                if str(indx)+question['id'] not in labels:
                    labels.append(str(indx)+question['id'])
        
        indx += 1
        print(indx)
    else:
        break


        

words = [stemmer.stem(w.lower()) for w in words if w not in discard]

words = sorted(list(set(words)))
labels = sorted(labels)

training, output = [], []
out_empty = [0 for _ in enumerate(labels)]

print(len(docx))
for x, doc in enumerate(docx):
    print(x)
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds: bag.append(1)
        else: bag.append(0)
    
    output_row = out_empty[:]
    output_row[labels.index(docy[x])] = 1

    training.append(bag)
    output.append(output_row)

print('turning to np training')
training = np.array(training)
print('turning to np output')
output = np.array(output)

print('dumpng')
with open('squad_data.pickle', 'wb') as f:
    pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save('./modelp/model.tflearn')