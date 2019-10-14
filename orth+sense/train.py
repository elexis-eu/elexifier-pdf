import pycrfsuite
from sklearn.metrics import confusion_matrix,classification_report
import argparse
def contextualise(feats):
  contextualised=[]
  for idx,feat in enumerate(feats):
    contextualised.append(feat[:])
    if idx>0:
      contextualised[-1].extend([e+'-1' for e in feats[idx-1]])
    if idx>1:
      contextualised[-1].extend([e+'-2' for e in feats[idx-2]])
    if idx>2:
      contextualised[-1].extend([e+'-3' for e in feats[idx-3]])
    if idx+2<len(feats):
      contextualised[-1].extend([e+'+1' for e in feats[idx+1]])
    if idx+3<len(feats):
      contextualised[-1].extend([e+'+2' for e in feats[idx+2]])
    if idx+4<len(feats):
      contextualised[-1].extend([e+'+3' for e in feats[idx+3]])
  return contextualised
parser=argparse.ArgumentParser()
parser.add_argument('-p', '--percentage', help='percentage of the training data to be used', type=float, default=1.0)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()
true=[]
pred=[]
import json
trainer=pycrfsuite.Trainer(algorithm='pa',verbose=False)
trainer.set_params({'max_iterations':40})
pages=json.load(open('train.json'))
for page in pages[:int(len(pages)*args.percentage)]:
  labels,feats=page
  trainer.append(contextualise(feats),labels)
trainer.train('model')
tagger=pycrfsuite.Tagger()
tagger.open('model')
pages=json.load(open('test.json'))
for page in pages:
  labels,feats=page
  pred_labels=tagger.tag(contextualise(feats))
  true.extend(labels)
  pred.extend(pred_labels)
  if args.verbose:
    print
    for f,l,p in zip(feats,labels,pred_labels):
      if p!=l:
        print '!',
      print p,l,' | '.join(f).encode('utf8')

print confusion_matrix(true,pred)
print classification_report(true,pred)
