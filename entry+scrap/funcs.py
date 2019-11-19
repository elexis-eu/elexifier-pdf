import numpy as np


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