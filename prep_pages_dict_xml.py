import xml.etree.ElementTree as ET
import json
import os

dict_xml = 'path/to/Bridge_A-AG_clean_aligned.xml'
tree_dict = ET.parse(dict_xml)
root_dict = tree_dict.getroot()
tokens_dict = list(root_dict.iter('TOKEN'))

tokens_annot = []
annotated_dir = 'path/to/dir'
for i in range(310):
    entry_xml = os.path.join(annotated_dir, 'entry_{}.xml'.format(i))
    try:
        tree_entry = ET.parse(entry_xml)
        parent_map = {c:p for p in tree_entry.iter() for c in p}
        root_entry = tree_entry.getroot()
        tokens_cur = list(root_entry.iter('TOKEN'))
        for idx, token in enumerate(tokens_cur):
            parent = parent_map[token]
            token.attrib['parent'] = parent.tag
        tokens_annot += tokens_cur
    except:
        print("error in file " + entry_xml)
        continue

pages_data = []
feats = []
labels = []
page_n = 1
for idx, token in enumerate(tokens_dict):

    page_t = int(token.attrib['page'])
    if page_t != page_n:
        pages_data.append((feats,labels))
        feats = []
        labels = []
        page_n = page_t

    if not token.text == tokens_annot[idx].text:
        print("misaligned data at idx {}! Breaking...".format(idx))
        break

    feat = ['SIZE='+token.attrib['font-size'],'BOLD='+token.attrib['bold'],'ITALIC='+token.attrib['italic'],'FONT='+token.attrib['font-name'],'TOKEN='+token.text]
    # TODO add newlines and such

    parent = tokens_annot[idx].attrib['parent']
    if parent == "dictScrap":
        label = "SCRAP"
    else:
        label = "ENTRY"

    feats.append(feat)
    labels.append(label)

json_file = 'path/to/pages_data.json'
json.dump(pages_data,open(json_file,'w'),indent=4)

