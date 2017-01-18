
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import load_model

tf.python.control_flow_ops = tf

df = pd.read_csv('listserve-archive.csv')

text = df.content.str.lower().str.cat(sep='\n\n--------\n\n', na_rep='')

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

model = load_model('hello_listserve.h5')

seed_string = "hello"
print("seed string -->", seed_string)
print('The generated text is')

out = seed_string
for i in range(320):
    x = np.zeros((1, len(seed_string), len(chars)))
    for t, char in enumerate(seed_string):
        x[0, t, char_indices[char]] = 1.
    preds = model.predict(x, verbose=0)[0]

    next_index = np.argmax(preds[len(seed_string)-1])
    #print('---', next_index, indices_char[next_index])

    next_char = indices_char[next_index]
    seed_string = seed_string + next_char

    out += next_char

print(out)
