import json
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Masking, Dense, Bidirectional, LSTM, TimeDistributed, Concatenate
import argparse
from datetime import datetime
from time import time
import os

from funcs import extract_tokens, one_hot_and_chars, one_hot_target



parser=argparse.ArgumentParser()
parser.add_argument('-p', '--percentage', help='percentage of the training data to be used', type=float, default=1.0)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-nr', '--n_rounds', help='number of repetitions of 10 epoch training', type=int, default=10)
parser.add_argument('-ld', '--logdir', help='directory where the log file will be stored', type=str, default="")
args = parser.parse_args()

t0 = time()
dt = datetime.now().strftime( "%Y%m%d-%H%M%S" )
logfile = None
if args.logdir is not "":
  if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
  logfile = os.path.join(args.logdir, "train__c-lstm_{}.log".format(dt))


train = json.load(open('entries_data_train.json', 'r'))
test = json.load(open('entries_data_test.json', 'r'))

x_train = [e[0] for e in train]
y_train = [e[1] for e in train]
x_test = [e[0] for e in test]
y_test = [e[1] for e in test]


all_tokens, idx_t, idx_c = extract_tokens(x_train)
x_train_oh, x_train_chars, idx = one_hot_and_chars(x_train, idx_c=idx_c)
x_test_oh, x_test_chars, _ = one_hot_and_chars(x_test, idx, idx_c)

max_sequence_len=max([len(e) for e in x_train_oh])*2
x_train_oh=sequence.pad_sequences(x_train_oh,max_sequence_len)
x_test_oh=sequence.pad_sequences(x_test_oh,max_sequence_len)

max_carray_len = 0
for seq in x_train_chars:
  max_cur = max([len(arr) for arr in seq])
  if max_cur > max_carray_len:
    max_carray_len = max_cur
max_carray_len *= 2

x_train_chars_pad = [sequence.pad_sequences(seq, max_carray_len) for seq in x_train_chars]
x_train_chars = sequence.pad_sequences(x_train_chars_pad, max_sequence_len)
x_test_chars_pad = [sequence.pad_sequences(seq, max_carray_len) for seq in x_test_chars]
x_test_chars = sequence.pad_sequences(x_test_chars_pad, max_sequence_len)

print("X_train_oh.shape", x_train_oh.shape)
print("X_train_chars.shape", x_train_chars.shape)
print("X_test_oh.shape", x_test_oh.shape)
print("X_test_chars.shape", x_test_chars.shape)

y_train_oh,idx_label=one_hot_target(y_train)
y_test_oh,_=one_hot_target(y_test,idx_label)
y_train_oh=sequence.pad_sequences(y_train_oh,max_sequence_len)
y_test_oh=sequence.pad_sequences(y_test_oh,max_sequence_len)
rev_idx_label={v:k for k,v in idx_label.items()}


t1 = time()
print("Data preparation time:", (t1-t0), "s")

features_input = Input(shape=(max_sequence_len, len(idx)))
chars_input = Input(shape=(max_sequence_len, max_carray_len))
chars_masked = Masking(0) (chars_input)
features_masked = Masking(0) (features_input)
chars_embed = Bidirectional(LSTM(5, return_sequences=True)) (chars_masked)
# chars_embed = Dropout(0.5) (chars_embed)
features_merged = Concatenate(axis=-1) ([features_masked, chars_embed])

h = Bidirectional(LSTM(12, return_sequences=True)) (features_merged)
# h = Dropout(0.5) (h)
y = TimeDistributed(Dense(len(idx_label), activation='softmax')) (h)
model = Model(inputs=[features_input, chars_input], outputs=y)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


t1 = time()
for i_round in range(args.n_rounds):
  print('ROUND', i_round)
  t_r0 = time()
  model.fit([x_train_oh, x_train_chars], y_train_oh, batch_size=5, epochs=10, validation_data=([x_test_oh, x_test_chars], y_test_oh), shuffle=True)
  score, acc = model.evaluate([x_test_oh, x_test_chars], y_test_oh, batch_size=50)

  y_test_pred=model.predict([x_test_oh, x_test_chars])

  def to_tags(sequence):
    result=[]
    for label in sequence:
      result.append(rev_idx_label[np.argmax(label)])
    return result

  lf = None
  if logfile is not None:
    lf = open(logfile, "a+")
    lf.write("ROUND {}\n\n".format(i_round))

  pred=[]
  true=[]
  from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
  for p,t,f in zip(y_test_pred,y_test_oh,x_test):
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
