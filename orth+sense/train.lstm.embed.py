from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Dropout, TimeDistributed, InputLayer, Masking, Bidirectional
from keras.layers import LSTM, Input, Concatenate
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np
import argparse
from time import time
from datetime import datetime
import os
parser=argparse.ArgumentParser()
parser.add_argument('-p', '--percentage', help='percentage of the training data to be used', type=float, default=1.0)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-ld', '--logdir', help='directory where the log file will be stored', type=str, default="")
args = parser.parse_args()

t0 = time()
dt = datetime.now().strftime( "%Y%m%d-%H%M%S" )
logfile = None
if args.logdir is not "":
  if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
  logfile = os.path.join(args.logdir, "train_{}.log".format(dt))

train=json.load(open('train.json'))
train=train[:int(len(train)*args.percentage)]
X_train=[e[1] for e in train]
y_train=[e[0] for e in train]
test=json.load(open('test.json'))
X_test=[e[1] for e in test]
y_test=[e[0] for e in test]

#max_lexicon=100

def extract_tokens(dataset):
  tokens = set()
  chars = set()
  for sequence in dataset:
    for instance in sequence:
      for feature in instance:
        if feature.startswith('TOKEN'):
          tokens.add(feature)
          for c in feature[6:]:
            chars.add(c)
  idx_t = {f:(i+1) for i, f in enumerate(tokens)}
  idx_c = {c:(i+1) for i, c in enumerate(chars)}
  return tokens, idx_t, idx_c

def one_hot(dataset, idx={}, idx_t={}):
  if len(idx)==0:
    lexicon=set()
    for sequence in dataset:
      for instance in sequence:
        for feature in instance:
          if not feature.startswith('TOKEN'):
            lexicon.add(feature)
    idx = {f:i for i,f in enumerate(lexicon)}
  oh_dataset=[]
  tokens_dataset = []
  for sequence in dataset:
    oh_sequence=[]
    token_sequence = []
    for instance in sequence:
      oh_instance=np.zeros(len(idx))
      for feature in instance:
        if feature in idx:
          oh_instance[idx[feature]]=1
          continue
        if len(idx_t)>0 and feature.startswith('TOKEN'):
          if feature in idx_t.keys():
            token_sequence.append(idx_t[feature])
          else:
            token_sequence.append(0)

      oh_sequence.append(oh_instance)
    oh_dataset.append(oh_sequence)
    tokens_dataset.append(token_sequence)
  return oh_dataset, tokens_dataset, idx

def one_hot_and_chars(dataset, idx={}, idx_c={}):
  if len(idx)==0:
    lexicon=set()
    for sequence in dataset:
      for instance in sequence:
        for feature in instance:
          if not feature.startswith('TOKEN'):
            lexicon.add(feature)
    idx = {f:i for i,f in enumerate(lexicon)}
  oh_dataset=[]
  chars_dataset = []
  for sequence in dataset:
    oh_sequence=[]
    char_arrays_sequence = []
    for instance in sequence:
      oh_instance=np.zeros(len(idx))
      char_array = []
      for feature in instance:
        if feature in idx:
          oh_instance[idx[feature]]=1
          continue
        if len(idx_c)>0 and feature.startswith('TOKEN'):
          for c in feature[6:]:
            if c in idx_c.keys():
              char_array.append(idx_c[c])
            else:
              char_array.append(0)

      oh_sequence.append(oh_instance)
      char_arrays_sequence.append(char_array)
    oh_dataset.append(oh_sequence)
    chars_dataset.append(char_arrays_sequence)
  return oh_dataset, chars_dataset, idx

all_tokens, idx_t, _ = extract_tokens(X_train)
X_train_oh, X_train_tokens, idx = one_hot(X_train, idx_t=idx_t)
X_test_oh, X_test_tokens, _ = one_hot(X_test, idx, idx_t)
# all_tokens, idx_t, idx_c = extract_tokens(X_train)
# X_train_oh, X_train_chars, idx = one_hot_and_chars(X_train, idx_c=idx_c)
# X_test_oh, X_test_chars, _ = one_hot_and_chars(X_test, idx, idx_c)
#print to_categorical(train[0][1])



def one_hot_target(dataset,idx={}):
  if len(idx)==0:
    lexicon=set()
    for sequence in dataset:
      for label in sequence:
        lexicon.add(label)
    idx={l:i for i,l in enumerate(lexicon)}
  cat_dataset=[]
  for sequence in dataset:
    cat_sequence=[]
    for label in sequence:
      oh_label=np.zeros(len(idx))
      oh_label[idx[label]]=1
      cat_sequence.append(oh_label)
    cat_dataset.append(cat_sequence)
  return np.array(cat_dataset),idx

max_sequence_len=max([len(e) for e in X_train_oh])*2
#print max_sequence_len
X_train_oh=sequence.pad_sequences(X_train_oh,max_sequence_len)
X_train_tokens = sequence.pad_sequences(X_train_tokens, max_sequence_len)

# max_carray_len = 0
# for seq in X_train_chars:
#   max_cur = max([len(arr) for arr in seq])
#   if max_cur > max_carray_len:
#     max_carray_len = max_cur
# max_carray_len *= 2
# X_train_chars_pad = [sequence.pad_sequences(seq, max_carray_len) for seq in X_train_chars]
# X_train_chars = sequence.pad_sequences(X_train_chars_pad, max_sequence_len)

X_test_nonpad=X_test_oh[:]
X_test_oh=sequence.pad_sequences(X_test_oh,max_sequence_len)
X_test_tokens = sequence.pad_sequences(X_test_tokens, max_sequence_len)
# X_test_chars_pad = [sequence.pad_sequences(seq, max_carray_len) for seq in X_test_chars]
# X_test_chars = sequence.pad_sequences(X_test_chars_pad, max_sequence_len)

print("X_train_oh.shape", X_train_oh.shape)
print("X_train_tokens.shape", X_train_tokens.shape)
# print("X_train_chars.shape", X_train_chars.shape)
print("X_test_oh.shape", X_test_oh.shape)
print("X_test_tokens.shape", X_test_tokens.shape)
# print("X_test_chars.shape", X_test_chars.shape)
y_train_oh,idx_label=one_hot_target(y_train)
y_test_oh,_=one_hot_target(y_test,idx_label)
y_train_oh=sequence.pad_sequences(y_train_oh,max_sequence_len)
y_test_oh=sequence.pad_sequences(y_test_oh,max_sequence_len)
rev_idx_label={v:k for k,v in idx_label.items()}
#print list(y_train_oh[0])

t1 = time()
print("Data preparation time:", (t1-t0), "s")

# # no embedding
# model = Sequential()
# #model.add(InputLayer(input_shape=(max_sequence_len,len(idx),)))
# #model.add(Embedding(len(idx),16))
# model.add(Masking(input_shape=(max_sequence_len,len(idx))))
# #model.add(Dropout(0.5))
# model.add(Bidirectional(LSTM(12, input_shape=(max_sequence_len,len(idx)),recurrent_dropout=0.0,dropout=0.0,return_sequences=True)))
# model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(len(idx_label),activation='softmax')))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# embedded tokens
tokens_input = Input(shape=(max_sequence_len,))
features_input = Input(shape=(max_sequence_len, len(idx)))
tokens_masked = Masking(0) (tokens_input)
features_masked = Masking(0) (features_input)
tokens_embed = Embedding(int((2/3)*len(idx_t)), 10, input_length=max_sequence_len) (tokens_masked)
tokens_embed = Dropout(0.5) (tokens_embed)
features_merged = Concatenate(axis=-1) ([features_masked, tokens_embed])

# features_input = Input(shape=(max_sequence_len, len(idx)))
# chars_input = Input(shape=(max_sequence_len, max_carray_len))
# chars_masked = Masking(0) (chars_input)
# features_masked = Masking(0) (features_input)
# chars_embed = Bidirectional(LSTM(5, return_sequences=True)) (chars_masked)
# # chars_embed = Dropout(0.5) (chars_embed)
# features_merged = Concatenate(axis=-1) ([features_masked, chars_embed])

h = Bidirectional(LSTM(12, return_sequences=True)) (features_merged)
# h = Dropout(0.5) (h)
y = TimeDistributed(Dense(len(idx_label), activation='softmax')) (h)
model = Model(inputs=[features_input, tokens_input], outputs=y)
# model = Model(inputs=[features_input, chars_input], outputs=y)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

n_rounds=10
for i in range(n_rounds):
  print('ROUND', i)
  t_r0 = time()
  # model.fit(X_train_oh, y_train_oh, batch_size=4, epochs=10, validation_data=(X_test_oh, y_test_oh), shuffle=True)
  # score, acc = model.evaluate(X_test_oh, y_test_oh, batch_size=50)
  model.fit([X_train_oh, X_train_tokens], y_train_oh, batch_size=4, epochs=10, validation_data=([X_test_oh, X_test_tokens], y_test_oh), shuffle=True)
  score, acc = model.evaluate([X_test_oh,X_test_tokens], y_test_oh, batch_size=50)
  # model.fit([X_train_oh, X_train_chars], y_train_oh, batch_size=5, epochs=10, validation_data=([X_test_oh, X_test_chars], y_test_oh), shuffle=True)
  # score, acc = model.evaluate([X_test_oh,X_test_chars], y_test_oh, batch_size=50)


  # y_test_pred=model.predict(X_test_oh)
  y_test_pred=model.predict([X_test_oh, X_test_tokens])
  # y_test_pred=model.predict([X_test_oh, X_test_chars])

  def to_tags(sequence):
    result=[]
    for label in sequence:
      result.append(rev_idx_label[np.argmax(label)])
    return result

  lf = None
  if logfile is not None:
    lf = open(logfile, "a+")
    lf.write("ROUND {}\n\n".format(i))

  pred=[]
  true=[]
  from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
  for p,t,f in zip(y_test_pred,y_test_oh,X_test):
    for i,u in enumerate(t):
      if sum(u)>0:
        break
    if args.verbose:
      for a,b,c in zip(to_tags(p[i:]),to_tags(t[i:]),f):
        if a!=b:
          # print('!',a,b,'|'.join(c))
          if lf: lf.write('! {} {} {}\n'.format(a, b, '|'.join(c)))
        else:
          # print(a,b,'|'.join(c))
          if lf: lf.write('{} {} {}\n'.format(a, b, '|'.join(c)))
      # print("")
      if lf: lf.write("\n")
    pred.extend(to_tags(p[i:]))
    true.extend(to_tags(t[i:]))
  print(confusion_matrix(true,pred))
  print(classification_report(true,pred))
  print("Accuracy:", accuracy_score(true,pred))
  print("ROUND time:", (time()-t_r0), "s")
  print("")
  if lf:
    lf.write("{}\n".format(confusion_matrix(true,pred)))
    lf.write("{}\n".format(classification_report(true,pred)))
    lf.write("Accuracy: {}\n".format(accuracy_score(true,pred)))
    lf.write("ROUND time: {} s\n".format((time()-t_r0)))
    lf.write("\n")
    lf.close()


print("Training time:", (time()-t1), "s")