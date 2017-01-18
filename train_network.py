import sys
import threading
import time

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.layers import (LSTM, Activation, Dense, Dropout, SimpleRNN,
                          TimeDistributed)
from keras.models import Sequential, load_model

tf.python.control_flow_ops = tf

df = pd.read_csv('listserve-archive.csv')


text = df.content.str.lower().str.cat(sep='\n\n--------\n\n', na_rep='')

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

# TODO we want to run enough epochs that we iterate this dataset several times...
def count_sentences():
    maxlen = 40
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen+1, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i+1:i +1+ maxlen])
        #if i<10 :
           # print (text[i: i + maxlen])
    	#print(text[i+1:i +1+ maxlen])
    print('nb sequences:', len(sentences))

@threadsafe_generator
def char_rnn_input():
    maxlen = 40
    step = 1

    counter = 0
    max_counter = 128*5
    
    X = np.zeros((max_counter, maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((max_counter, maxlen, len(chars)), dtype=np.bool)
    
    while True:
        print('New iteration of corpus')
        for i in range(0, len(text) - maxlen+1, step):
            start, end = i, i+maxlen
            sentence = text[start:end]
            next_chars = text[start+1:end+1]

            for t, (char, next_char) in enumerate(zip(sentence, next_chars)):
                X[counter, t, char_indices[char]] = 1
                y[counter, t, char_indices[next_char]] = 1

            counter += 1

            if counter == max_counter:
                counter = 0
                X = np.zeros((max_counter, maxlen, len(chars)), dtype=np.bool)
                y = np.zeros((max_counter, maxlen, len(chars)), dtype=np.bool)
                yield X, y 

# We want to exhaust the corpus at least once
# it seems like we can do that by running more
# than (num_sentences / max_counter) < 8000 epochs
count_sentences()
#sys.exit()

print('Build model...')

model = load_model('hello_listserve.h5')

# model = Sequential()
# model.add(LSTM(512, input_dim=len(chars), return_sequences=True))
# model.add(LSTM(512, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(TimeDistributed(Dense(len(chars))))
# model.add(Activation('softmax'))
# 
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print(model.summary())

generator = char_rnn_input()

t0=time.time()
model.fit_generator(
    generator,
    samples_per_epoch=128,
    nb_epoch=50000,
    #nb_epoch=1000*8,
    verbose=1,
    validation_data=None,
    class_weight=None,
    pickle_safe=True,
    nb_worker=2
)
t1=time.time()

print("Training completed in " + str(t1-t0) + " seconds")

model.save('hello_listserve.h5')
print('-- training')

seed_string="hello"
print("seed string -->", seed_string)
print('The generated text is')

out = seed_string
for i in range(320):
    x=np.zeros((1, len(seed_string), len(chars)))
    for t, char in enumerate(seed_string):
        x[0, t, char_indices[char]] = 1.
    preds = model.predict(x, verbose=0)[0]

    next_index=np.argmax(preds[len(seed_string)-1])
    #print('---', next_index, indices_char[next_index])

    next_char = indices_char[next_index]
    seed_string = seed_string + next_char
    
    out += next_char

print(out)
