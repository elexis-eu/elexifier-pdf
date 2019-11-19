import json
import numpy as np
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Masking, Dense, Bidirectional, LSTM, TimeDistributed

from funcs import one_hot, one_hot_target


train = json.load(open('train.json', 'r'))
test = json.load(open('test.json', 'r'))
x_train = [e[0] for e in train]
y_train = [e[1] for e in train]
x_test = [e[0] for e in test]
y_test = [e[1] for e in test]



x_train_oh,idx=one_hot(x_train)
x_test_oh,_=one_hot(x_test,idx)

max_sequence_len=max([len(e) for e in x_train_oh])*2
x_train_oh=sequence.pad_sequences(x_train_oh,max_sequence_len)
x_test_oh=sequence.pad_sequences(x_test_oh,max_sequence_len)

y_train_oh,idx_label=one_hot_target(y_train)
y_test_oh,_=one_hot_target(y_test,idx_label)
y_train_oh=sequence.pad_sequences(y_train_oh,max_sequence_len)
y_test_oh=sequence.pad_sequences(y_test_oh,max_sequence_len)
rev_idx_label={v:k for k,v in idx_label.items()}


model = Sequential()
model.add(Masking(input_shape=(max_sequence_len,len(idx))))
model.add(Bidirectional(LSTM(12, input_shape=(max_sequence_len,len(idx)),recurrent_dropout=0.0,dropout=0.0,return_sequences=True)))
model.add(TimeDistributed(Dense(len(idx_label),activation='softmax')))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
t1 = time()
n_rounds=10
for i_round in range(n_rounds):
  print('ROUND', i_round)
  t_r0 = time()
  model.fit(x_train_oh, y_train_oh, batch_size=4, epochs=10, validation_data=(x_test_oh, y_test_oh), shuffle=True)
  score, acc = model.evaluate(x_test_oh, y_test_oh, batch_size=50)

  y_test_pred=model.predict(x_test_oh)

  def to_tags(sequence):
    result=[]
    for label in sequence:
      result.append(rev_idx_label[np.argmax(label)])
    return result

  lf = None
  # if logfile is not None:
  #   lf = open(logfile, "a+")
  #   lf.write("ROUND {}\n\n".format(i))

  pred=[]
  true=[]
  from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
  for p,t,f in zip(y_test_pred,y_test_oh,x_test):
    for i,u in enumerate(t):
      if sum(u)>0:
        break
    if True: #args.verbose:
      for a,b,c in zip(to_tags(p[i:]),to_tags(t[i:]),f):
        if a!=b:
          print('!',a,b,'|'.join(c))
          # if lf: lf.write('! {} {} {}\n'.format(a, b, '|'.join(c)))
        else:
          print(a,b,'|'.join(c))
          # if lf: lf.write('{} {} {}\n'.format(a, b, '|'.join(c)))
      print("")
      # if lf: lf.write("\n")
    pred.extend(to_tags(p[i:]))
    true.extend(to_tags(t[i:]))
  print(confusion_matrix(true,pred))
  print(classification_report(true,pred))
  print("Accuracy:", accuracy_score(true,pred))
  print("ROUND time:", (time()-t_r0), "s")
  print("")
  # if lf:
  #   lf.write("{}\n".format(confusion_matrix(true,pred)))
  #   lf.write("{}\n".format(classification_report(true,pred)))
  #   lf.write("Accuracy: {}\n".format(accuracy_score(true,pred)))
  #   lf.write("ROUND time: {} s\n".format((time()-t_r0)))
  #   lf.write("\n")
  #   lf.close()


print("Training time:", (time()-t1), "s")
