import xml.etree.ElementTree as ET
import json


def json2xml( json_in_file, xml_raw, xml_out_file ):

    json_data = json.load( open( json_in_file, 'r' ) )

    tree_raw = ET.parse( xml_raw )
    root_raw = tree_raw.getroot()
    tokens_raw = list( root_raw.iter( 'TOKEN' ) )

    page_level_tokens = json_data['page_level'][0]
    page_level_labels = json_data['page_level'][1]
    entry_level_tokens = json_data['entry_level'][0]
    entry_level_labels = json_data['entry_level'][1]
    sense_level_tokens = json_data['sense_level'][0]
    sense_level_labels = json_data['sense_level'][1]


    # get label for each token in the raw file
    token_labels = []

    i_p = 0
    i_e = 0
    i_s = 0
    i_pt = 0
    i_et = 0
    i_st = 0
    prev_page = int( tokens_raw[0].attrib['page'] )
    for token_r in tokens_raw:

        label_cur = ["", "", ""]

        if token_r.text is None:  # empty tokens were skipped
            continue

        cur_page = int( token_r.attrib['page'] )

        if prev_page != cur_page:
            i_p += 1
            i_pt = 0
            prev_page = cur_page

        token_lvl1 = page_level_tokens[i_p][i_pt]
        token_lvl1_text = token_lvl1[4][6:]

        if token_r.text != token_lvl1_text:
            print( "error! misaligned data!" )
            break

        label_cur[0] = page_level_labels[i_p][i_pt]

        if i_e < len( entry_level_tokens ):
            token_lvl2 = entry_level_tokens[i_e][i_et]
            token_lvl2_text = token_lvl2[4][6:]

            if token_r.text == token_lvl2_text:
                label_cur[1] = entry_level_labels[i_e][i_et]
                i_et += 1
                if i_et == len( entry_level_tokens[i_e] ):
                    i_e += 1
                    i_et = 0

                if i_s < len( sense_level_tokens ):

                    token_lvl3 = sense_level_tokens[i_s][i_st]
                    token_lvl3_text = token_lvl3[4][6:]

                    if token_r.text == token_lvl3_text:
                        label_cur[2] = sense_level_labels[i_s][i_st]
                        i_st += 1
                        if i_st == len( sense_level_labels[i_s] ):
                            i_s += 1
                            i_st = 0

        token_labels.append( label_cur )
        i_pt += 1

    # # doc_elm = ET.Element( 'document' )
    body_elm = ET.Element( 'body' )
    cur_elm_lvl1 = None
    cur_elm_lvl2 = None
    cur_elm_lvl3 = None
    labels_prev = ['', '', '']

    for i, token_r in enumerate( tokens_raw ):

        if i > 0: labels_prev = token_labels[i-1]

        lbl_lvl1 = token_labels[i][0]
        lbl_lvl2 = token_labels[i][1]
        lbl_lvl3 = token_labels[i][2]

        if lbl_lvl1 == 'SCRAP':
            if cur_elm_lvl1 is not None and cur_elm_lvl1.tag == 'container' and cur_elm_lvl1.attrib['name'] == 'dictScrap':
                cur_elm_lvl1.append( token_r )
                continue
            else:
                cur_elm_lvl1 = ET.SubElement( body_elm, 'container', attrib={'name' : 'dictScrap'} )
                cur_elm_lvl1.append( token_r )
                continue

        elif 'ENTRY' in lbl_lvl1:

            if lbl_lvl1 == 'ENTRY_START' or cur_elm_lvl1 is None or (cur_elm_lvl1.tag == 'container' and cur_elm_lvl1.attrib['name'] != 'entry'):
                cur_elm_lvl1 = ET.SubElement( body_elm, 'container', attrib={'name': 'entry'} )

            if lbl_lvl2 == 'INSIDE':
                cur_elm_lvl1.append( token_r )
                continue

            elif lbl_lvl2 == 'FORM':
                if labels_prev[1] != 'FORM':
                    cur_elm_lvl2 = ET.SubElement( cur_elm_lvl1, 'container', attrib={'name': 'form'} )
                cur_elm_lvl2.append( token_r )
                continue

            elif lbl_lvl2 == 'POS':
                if labels_prev[1] != 'POS':
                    cur_elm_lvl2 = ET.SubElement( cur_elm_lvl1, 'container', attrib={'name': 'pos'} )
                cur_elm_lvl2.append( token_r )
                continue

            elif 'SENSE' in lbl_lvl2:

                if lbl_lvl2 == 'SENSE_START' or cur_elm_lvl2 is None or (cur_elm_lvl2.tag == 'container' and cur_elm_lvl2.attrib['name'] != 'sense'):
                    cur_elm_lvl2 = ET.SubElement( cur_elm_lvl1, 'container', attrib={'name': 'sense'} )

                if lbl_lvl3 == 'INSIDE':
                    cur_elm_lvl2.append( token_r )
                    continue
                elif lbl_lvl3 == 'TRANS':
                    if labels_prev[2] != 'TRANS':
                        cur_elm_lvl3 = ET.SubElement( cur_elm_lvl2, 'container', attrib={'name': 'trans'} )
                    cur_elm_lvl3.append( token_r )




    # label_prev = ""
    # i_p = 0
    # i_e = 0
    # i_s = 0
    # i_pt = 0
    # prev_page = int( tokens_raw[0].attrib['page'] )
    # for token_r in tokens_raw:
    #
    #     if token_r.text is None:        # empty tokens were skipped
    #         continue
    #
    #     cur_page = int( token_r.attrib['page'] )
    #
    #     if prev_page != cur_page:
    #         i_p += 1
    #         i_pt = 0
    #         prev_page = cur_page
    #
    #     token_lvl1 = page_level_tokens[i_p][i_pt]
    #     token_lvl1_text = token_lvl1[4][6:]
    #
    #     if token_r.text != token_lvl1_text:
    #         print( "error! misaligned data!" )
    #         break
    #
    #     label_cur = page_level_labels[i_p][i_pt]
    #
    #     if label_cur == 'SCRAP':
    #         if cur_elm.tag == 'container' and cur_elm.attrib['name'] == 'dictScrap':
    #             cur_elm.append( token_r )
    #         else:
    #             scrap_cont = ET.SubElement( cur_elm, 'container', attrib={'name' : 'dictScrap'} )
    #             cur_elm = scrap_cont
    #             cur_elm.append( token_r )
    #
    #     elif label_cur == 'ENTRY_START':
    #         entry_cont = ET.SubElement( body_elm, 'container', attrib={'name' : 'entry'} )
    #         cur_elm = entry_cont
    #         cur_elm.append( token_r )
    #
    #         # first element of entry level predictions should be here
    #         token_lvl2 = entry_level_tokens[i_e][0]
    #         label_lvl2 = entry_level_labels[i_e][0]
    #         cur_elm_lvl1 = cur_elm
    #         if label_lvl2 == "FORM":
    #             form_cont = ET.SubElement( cur_elm_lvl1, 'container', attrib={'name': 'form'} )
    #         elif label_lvl2 == 'POS':
    #             pos_cont = ET.SubElement( cur_elm_lvl1, 'container', attrib={'name': 'pos'} )
    #
    #
    #     elif label_cur == 'ENTRY_INSIDE':
    #         if cur_elm.tag != 'container' or (cur_elm.tag == 'container' and cur_elm.attrib['name'] != 'entry'):
    #             entry_cont = ET.SubElement( body_elm, 'container', attrib={'name': 'entry'} )
    #             cur_elm = entry_cont
    #
    #         # TODO implement other levels, ENTRY and SENSE levels here
    #
    #         cur_elm.append( token_r )
    #
    #     label_prev = label_cur
    #     i_pt += 1


    xml_string = ET.tostring( body_elm, encoding='unicode', method='xml' )
    with open( xml_out_file, 'w' ) as f:
        f.write( xml_string )

    return xml_string



if __name__ == "__main__":

    json_ml_results_file = '/media/jan/Fisk/CJVT/outputs/json/predicted_data/mali_sloang_trained_4.json'
    xml_raw_file = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/mali_sloang_pred_prelomom-20-pages.xml'
    xml_out_file = '/media/jan/Fisk/CJVT/outputs/json/predicted_data/mali_sloang_out_new.xml'

    json2xml( json_ml_results_file, xml_raw_file, xml_out_file )


