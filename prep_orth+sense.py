import xml.etree.ElementTree as ET
import json
train=[]
test=[]
def are_related(e1,e2):
  while True:
    if e1 in e2:
      return True
    if e1 in parent_map:
      e1=parent_map[e1]
    else:
      return False
    
for i in range(310):
  if i<200:
    add=train
  else:
    add=test
  try:
    tree=ET.parse('tokenizer/dictScrap/entry_'+str(i)+'.xml')
  except:
    continue
  root=tree.getroot()
  parent_map={c:p for p in tree.iter() for c in p}
  tokens=list(root.iter('TOKEN'))
  orth=None
  sense=None
  feats=[]
  labels=[]
  prev_base=0
  for element in root.iter():
    if element.tag=='orth':
      orth=element
    if element.tag=='sense':
      sense=element
    if element.tag=='TOKEN':
      label=None
      if element!=None:
        if element in orth:
          label='ORTH'
      if sense!=None:
        if are_related(element,sense):
          label='SENSE'+sense.attrib.get('n','1')
      if label==None:
        label='O'
      token=element
      if float(token.attrib['base'])!=prev_base:
        #labels.append('NEWLINE')
        #feats.append(['NEWLINE'])
        newline=True
        prev_base=float(token.attrib['base'])
      else:
        newline=False
      if token.text==None:
        token.text=''
      feat=['SIZE='+token.attrib['font-size'],'BOLD='+token.attrib['bold'],'ITALIC='+token.attrib['italic'],'FONT='+token.attrib['font-name'],'TOKEN='+token.text]
      if newline:
        feat.append('NEWLINE')
      feats.append(feat)
      labels.append(label)
  feats[0].append('BOS')
  feats[-1].append('EOS')
  sense=''
  for idx,label in enumerate(labels):
    if label.startswith('SENSE'):
      if sense!=label:
        labels[idx]='SENSE_START'
      else:
        labels[idx]='SENSE_INSIDE'
      #if idx+1<len(labels):
      #  if labels[idx+1]!=label:
      #    labels[idx]='SENSE_END'
      sense=label
  add.append((labels,feats))

json.dump(train,open('orth+sense/train.json','w'),indent=4)
json.dump(test,open('orth+sense/test.json','w'),indent=4)
