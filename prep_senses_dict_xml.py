import xml.etree.ElementTree as ET
import json
import os

def pos_def_or_quote(e, parent_map):
    while True:
        if e.tag.lower() == 'entry':
            return '-1'
        if parent_map[e].tag.lower() == 'sense':
            return '0'
        elif parent_map[e].tag.lower() == 'dictscrap':
            return '0'
        elif parent_map[e].tag.lower() == 'pos':
            return 'POS'
        elif parent_map[e].tag.lower() == 'def':
            return 'DEF'
        elif parent_map[e].tag.lower() == 'quote':
            return 'QUOTE'
        e = parent_map[e]



dict_xml = 'path/to/Bridge_A-AG_clean_aligned.xml'
tree_dict = ET.parse(dict_xml)
root_dict = tree_dict.getroot()
tokens_dict = list(root_dict.iter('TOKEN'))

annotated_dir = 'path/to/dir'
idx_dict_t = 0
senses_data = []
for i in range(310):
    entry_xml = os.path.join(annotated_dir, 'entry_{}.xml'.format(i))
    try:
        tree_entry = ET.parse(entry_xml)
        parent_map = {c:p for p in tree_entry.iter() for c in p}
        root_entry = tree_entry.getroot()
        tokens_cur = list(root_entry.iter('TOKEN'))

        feats = []
        labels = []

        for idx, token_e in enumerate(tokens_cur):

            if not token_e.text == tokens_dict[idx_dict_t].text:
                print("misaligned data in entry {}! Breaking...".format(entry_xml))
                break

            label = pos_def_or_quote(token_e, parent_map)
            if label == '-1':
                idx_dict_t += 1
                continue

            token = tokens_dict[idx_dict_t]
            feat = ['SIZE=' + token.attrib['font-size'], 'BOLD=' + token.attrib['bold'], 'ITALIC=' + token.attrib['italic'], 'FONT=' + token.attrib['font-name'], 'TOKEN=' + token.text]

            feats.append(feat)
            labels.append(label)
            idx_dict_t += 1

        senses_data.append((feats, labels))


    except:
        print("error in file " + entry_xml)
        continue

json_file = 'path/to/senses_data.json'
json.dump(senses_data, open(json_file, 'w'), indent=4)
