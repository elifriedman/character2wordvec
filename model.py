from make_dataset import Corpus
import numpy as np
from keras.models import Model, Sequential
from keras.layers.core import Dense, Masking, Activation, Dropout, Reshape, Permute
from keras.layers import LSTM 

def build_model(masking_value, Timesteps, Dimension, Num_Outputs):
    model = Sequential()
    model.add(Masking(mask_value = masking_value, input_shape=(Timesteps, Dimension)))
    model.add(LSTM(128))
    model.add(Dense(512))
    model.add(Activation("tanh"))
    model.add(Dense(512))
    model.add(Activation("tanh"))
    model.add(Dense(Num_Outputs))
    model.add(Activation("softmax"))
    model.compile("adam","categorical_crossentropy", metrics=['accuracy'])
    return model

import sys
def main(filenames):
    corpus = Corpus(filenames)
    num_samples = 0
    for doc_num in range(len(corpus)):
        num_samples += len(corpus.getWords(doc_num))
    print num_samples

    X = np.zeros((num_samples, corpus.T, corpus.D)) # dim = (0 , # chars in word, # chars)
    Y = np.zeros((num_samples, corpus.W))           # dim = (0 , # words in dataset)
    for doc_num in range(len(corpus)):
        Xdoc = corpus.docChars2OneHot(doc_num) # dim = (# words in doc, # chars in word, # chars)
        Ydoc = corpus.docWords2OneHot(doc_num) # dim = (# words in doc, # words in dataset)
        Xdoc = Xdoc[0:-1, :, :]
        Ydoc = Ydoc[1:, : ] # predict next word

        X = np.concatenate([X, Xdoc], axis=0)
        Y = np.concatenate([Y, Ydoc], axis=0)

    model = build_model(corpus.masking_value, corpus.T, corpus.D, Y.shape[1])
    hist = model.fit(X, Y, validation_split=0.3, shuffle=True, nb_epoch=100)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage: python " + sys.argv[0] + " <filename1> [<filenam2>, ...]"
        sys.exit(-1)
    main(sys.argv[1:])
