from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, TimeDistributed, InputLayer, Masking, Bidirectional
from keras.layers import LSTM
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-p', '--percentage', help='percentage of the training data to be used', type=float, default=1.0)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

train=json.load(open('train.json'))
train=train[:int(len(train)*args.percentage)]
X_train=[e[1] for e in train]
y_train=[e[0] for e in train]
test=json.load(open('test.json'))
X_test=[e[1] for e in test]
y_test=[e[0] for e in test]

#max_lexicon=100

def one_hot(dataset,idx={}):
  if len(idx)==0:
    lexicon=set()
    for sequence in dataset:
      for instance in sequence:
        for feature in instance:
          if not feature.startswith('TOKEN'):
            lexicon.add(feature)
    idx={f:i for i,f in enumerate(lexicon)}
  oh_dataset=[]
  for sequence in dataset:
    oh_sequence=[]
    for instance in sequence:
      oh_instance=np.zeros(len(idx))
      for feature in instance:
        if feature in idx:
          oh_instance[idx[feature]]=1
      oh_sequence.append(oh_instance)
    oh_dataset.append(oh_sequence)
  return oh_dataset,idx

X_train_oh,idx=one_hot(X_train)
X_test_oh,_=one_hot(X_test,idx)
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
X_test_nonpad=X_test_oh[:]
X_test_oh=sequence.pad_sequences(X_test_oh,max_sequence_len)
print X_train_oh.shape
print X_test_oh.shape
y_train_oh,idx_label=one_hot_target(y_train)
y_test_oh,_=one_hot_target(y_test,idx_label)
y_train_oh=sequence.pad_sequences(y_train_oh,max_sequence_len)
y_test_oh=sequence.pad_sequences(y_test_oh,max_sequence_len)
rev_idx_label={v:k for k,v in idx_label.iteritems()}
#print list(y_train_oh[0])

model = Sequential()
#model.add(InputLayer(input_shape=(max_sequence_len,len(idx),)))
#model.add(Embedding(len(idx),16))
model.add(Masking(input_shape=(max_sequence_len,len(idx))))
#model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(12, input_shape=(max_sequence_len,len(idx)),recurrent_dropout=0.0,dropout=0.0,return_sequences=True)))
model.add(TimeDistributed(Dense(len(idx_label),activation='softmax')))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print model.summary()


for i in range(10):
  print 'ROUND',i
  model.fit(X_train_oh,y_train_oh,batch_size=5,epochs=10,validation_data=(X_test_oh,y_test_oh))
  score,acc=model.evaluate(X_test_oh,y_test_oh,batch_size=50)



  y_test_pred=model.predict(X_test_oh)

  def to_tags(sequence):
    result=[]
    for label in sequence:
      result.append(rev_idx_label[np.argmax(label)])
    return result

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
          print '!',a,b,'|'.join(c)
        else:
          print a,b,'|'.join(c)
      print
    pred.extend(to_tags(p[i:]))
    true.extend(to_tags(t[i:]))
  print confusion_matrix(true,pred)
  print classification_report(true,pred)
  print accuracy_score(true,pred)