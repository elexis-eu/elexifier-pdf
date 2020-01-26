import os
import xml.etree.ElementTree as ET

# first extract the dictionary xml TOKENs
dict_xml = 'path/to//Bridge_A-AG_clean_aligned.xml'
tree_dict = ET.parse(dict_xml)
root_dict = tree_dict.getroot()
tokens_dict = list(root_dict.iter('TOKEN'))

# then extract all the annotated TOKENs
tokens_entries = []
annotated_dir = 'path/to/dir'
for i in range(310):
    entry_xml = os.path.join(annotated_dir, 'entry_{}.xml'.format(i))
    try:
        tree_entry = ET.parse(entry_xml)
        root_entry = tree_entry.getroot()
        tokens_entries += list(root_entry.iter('TOKEN'))
    except:
        print("error in file " + entry_xml)
        continue




len_d = len(tokens_dict)
len_e = len(tokens_entries)
i_d = 0
i_e = 0

# for i in range(irng):
while True:

    if tokens_dict[i_d].text == tokens_entries[i_e].text:

        # print( "{}: {} ----- {}: {}".format(i_d, tokens_dict[i_d].text, i_e, tokens_entries[i_e].text) )
        pass

    else:

        tkn_d_new = tokens_dict[i_d].text
        tkn_e_new = tokens_entries[i_e].text

        i_d0 = i_d
        i_e0 = i_e

        found = False
        for j_d in range(i_d0, len_d):
            if tokens_dict[j_d].text == tkn_e_new:
                i_d = j_d
                found = True
                break

        if found:

            print( "thrown out instances from dict_xml from {} to {}:".format(i_d0, i_d) )
            for k in range(i_d0, i_d):
                print("\t"+tokens_dict[k].text)

            # print("{}: {} ----- {}: {}".format(i_d, tokens_dict[i_d].text, i_e, tokens_entries[i_e].text))

        else:
            for j_e in range(i_e0, len_e):
                if tokens_entries[j_e].text == tkn_d_new:
                    i_e = j_e
                    found = True
                    break

            if found:

                print("thrown out instances from entries from {} to {}:".format(i_e0, i_e))
                for k in range(i_e0, i_e):
                    print("\t" + tokens_entries[k].text)

                print("{}: {} ----- {}: {}".format(i_d, tokens_dict[i_d].text, i_e, tokens_entries[i_e].text))

        if not found:
            print( "no continuation found, ending...")
            break

    i_d += 1
    i_e += 1

    if i_d == len_d or i_e == len_e:
        print("reached the end")
        break


