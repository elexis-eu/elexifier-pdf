import xml.etree.ElementTree as ET
import json
train=[]
test=[]
prev_page_id='p1'
prev_base=0
labels=[]
feats=[]
for i in range(310):
  if i<200:
    add=train
  else:
    add=test
  try:
    tree=ET.parse('tokenizer/correct_entries/entry_'+str(i)+'.xml')
  except:
    continue
  root=tree.getroot()
  tokens=list(root.iter('TOKEN'))
  for idx,token in enumerate(tokens):
    page_id=token.attrib['sid'].split('_')[0]
    if page_id!=prev_page_id and len(labels)>0:
      feats[0].append('BOS')
      feats[-1].append('EOS')
      add.append((labels,feats))
      labels=[]
      feats=[]
      prev_base=0
      prev_page_id=page_id
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
    if idx==0:
      label='START'
    elif idx+1==len(tokens):
      label='INSIDE'
    else:
      label='INSIDE'
    feats.append(feat)
    labels.append(label)
json.dump(train,open('entry/train.json','w'),indent=4)
json.dump(test,open('entry/test.json','w'),indent=4)
