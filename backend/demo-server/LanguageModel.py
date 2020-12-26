import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, GRU
from difflib import SequenceMatcher
from wordfreq import word_frequency

class LanguageModel:
  def __init__(self, corpus):

    self.passwords = []
    with open(corpus, 'r') as corpus:
      for password in corpus.readlines():
        password = re.sub('[^a-z ]+', '', password)
        password = re.sub('[ ]+', ' ', password)
        if len(password)<5:
          continue
        self.passwords.append(password)
    self.words = self.passwords.copy()

    lengths = np.array([len(p) for p in self.passwords])
    max_len = np.max(lengths)
    min_len = np.min(lengths)
    mean_len = int(round(np.mean(lengths)))
    #print('Longest: ', max_len, '\nShortest: ', min_len, '\nMean len: ', mean_len)

    self.passwords = [p.ljust(max_len) for p in self.passwords]

    self.NUM_PASSWORDS = len(self.passwords)
    self.SEQ_LENGTH = 1
    self.BATCH_SIZE = 512
    self.BATCH_CHARS = max_len
    self.LSTM_SIZE = 512
    self.LAYERS = 2

    self.char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(''.join(self.passwords))))) }
    self.idx_to_char = { i: ch for (ch, i) in self.char_to_idx.items() }
    self.vocab_size = len(self.char_to_idx)
    #print('Working on %d passwords (vocab %d)' % (self.NUM_PASSWORDS, self.vocab_size))

    # self.region_split = {
    #   1: ['Q', 'W', 'A', 'S', 'E', 'D'],
    #   2: ['E', 'R', 'D', 'F', 'S', 'T', 'A'],
    #   3: ['Z', 'X', 'S', 'D', 'C'],
    #   4: ['T', 'Y', 'U', 'F', 'G', 'H', 'J', 'R', 'I', 'K'],
    #   5: ['I', 'O', 'P', 'K', 'L', 'M', 'N', 'J', 'U', 'H'],
    #   6: ['C', 'V', 'B', 'N', ' ', 'X', 'M', 'P', 'L']
    # }
    self.region_split = {
      1: ['Q', 'W', 'A', 'S', 'E', 'D'],
      2: ['E', 'R', 'D', 'F', 'S', 'T', 'A'],
      3: ['Z', 'X', 'S', 'D', 'C'],
      4: ['T', 'Y', 'U', 'F', 'G', 'H', 'J'],
      5: ['I', 'O', 'P', 'K', 'L', 'M', 'N', 'J', 'U'],
      6: ['C', 'V', 'B', 'N', ' ', 'X', 'M', 'P', 'L']
    }

    def key_area(key):
      for region, keys in self.region_split.items():
        if key.upper() in keys:
          return region
      raise Exception('Unknown key' + str(key))

    self.region_idx = {reg:[self.char_to_idx[i.lower()] for i in self.region_split[reg]] for reg in self.region_split}

    return

  def read_batches(self, passwords):
    T = np.asarray([[self.char_to_idx[c] for c in p] for p in passwords], dtype=np.int32)
    X = np.zeros((self.BATCH_SIZE, self.SEQ_LENGTH, self.vocab_size))
    Y = np.zeros((self.BATCH_SIZE, self.SEQ_LENGTH, self.vocab_size))

    for i in range(0, self.BATCH_CHARS - self.SEQ_LENGTH - 1, self.SEQ_LENGTH):
        X[:] = 0
        Y[:] = 0
        for batch_idx in range(self.BATCH_SIZE):
            for j in range(self.SEQ_LENGTH):
                X[batch_idx, j, T[batch_idx,i]] = 1
                Y[batch_idx, j, T[batch_idx,i+1]] = 1

        yield X, Y

  def build_train_model(self):
    model = Sequential()
    model.add(GRU(self.LSTM_SIZE, return_sequences=True, batch_input_shape=(self.BATCH_SIZE, self.SEQ_LENGTH, self.vocab_size), stateful=True))
    model.add(Dropout(0.2))
    for l in range(self.LAYERS - 1):
        model.add(GRU(self.LSTM_SIZE, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    #model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Dense(self.vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

  def build_test_model(self):
    model = Sequential()
    model.add(GRU(self.LSTM_SIZE, return_sequences=True, batch_input_shape=(1, 1, self.vocab_size), stateful=True))
    model.add(Dropout(0.2))
    for l in range(self.LAYERS - 1):
        model.add(GRU(self.LSTM_SIZE, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    #model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Dense(self.vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

  def load_weights(self, model_path):
    self.test_model = self.build_test_model()
    self.test_model.load_weights(model_path)

  def generate(self, region_seq, seed):
    self.test_model.reset_states()

    queue = [seed]
    finished = []

    while not len(queue) == 0:
      sent = queue.pop()
      self.test_model.reset_states()
      sampled = [self.char_to_idx[c] for c in sent]
      for c in sent[:-1]:
        batch = np.zeros((1, 1, self.vocab_size))
        batch[0, 0, self.char_to_idx[c]] = 1
        self.test_model.predict_on_batch(batch)

      for reg in region_seq[len(sent):]:
        batch = np.zeros((1, 1, self.vocab_size))
        batch[0, 0, sampled[-1]] = 1
        softmax = self.test_model.predict_on_batch(batch)[0].ravel()
        softmax = [(i,softmax[i]) for i in self.region_idx[reg]]
        softmax.sort(key=lambda x:x[1], reverse=True)
        queue.append(''.join([self.idx_to_char[c] for c in sampled + [softmax[1][0]]]))
        queue.append(''.join([self.idx_to_char[c] for c in sampled + [softmax[2][0]]]))
        sampled.append(softmax[0][0])
      
      finished.append(''.join([self.idx_to_char[c] for c in sampled]))
    
    return finished

  def get_probable_words(self, region_seq):
    generated = []
    for key in self.region_split[region_seq[0]]:
      generated.extend(self.generate(region_seq, key.lower()))
    
    def similar_own(p, q):
      score=0
      for c1, c2 in zip(p,q):
        if c1==c2:
          score+=1
      return score/len(p)

    def autocorrect(q):
      closest = [0.65, []]
      l = len(q)
      for p in [word for word in self.words if len(word)==l]:
        simi = similar_own(p, q)
        if simi > closest[0]:
          closest[0] = simi
          closest[1] = [p]
        if simi == closest[0]:
          closest[1].append(p)
        if simi == 1: break
      return closest[1]
    
    def word_popularity(word):
      return word_frequency(word, 'en', wordlist='small')
    
    correct = []
    for g in generated:
      if g in self.words:
          correct.append(g)
          generated.remove(g)
    corrected = []
    for q in generated:
      corrected.extend(autocorrect(q))

    return correct + sorted(set(corrected), key=word_popularity, reverse=True)[:30]
