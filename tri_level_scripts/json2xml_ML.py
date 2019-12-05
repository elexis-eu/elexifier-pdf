import xml.etree.ElementTree as ET
import json


def json2xml( json_in_file, xml_raw, xml_out_file ):

    json_data = json.load( open( json_in_file, 'r' ) )

    tree_raw = ET.parse( xml_raw )
    root_raw = tree_raw.getroot()
    tokens_raw = list( root_raw.iter( 'TOKEN' ) )

    page_level_tokens = json_data['page_level'][0]
    page_level_labels = json_data['page_level'][1]
    # entry_level_tokens = json_data['entry_level'][0]
    # entry_level_labels = json_data['entry_level'][1]
    # sense_level_tokens = json_data['sense_level'][0]
    # sense_level_labels = json_data['sense_level'][1]

    doc_elm = ET.Element( 'document' )
    body_elm = ET.SubElement( doc_elm, 'body' )
    cur_elm = body_elm

    label_prev = ""
    i_p = 0
    # i_e = 0
    # i_s = 0
    i_pt = 0
    prev_page = int( tokens_raw[0].attrib['page'] )
    for token_r in tokens_raw:

        if token_r.text is None:        # empty tokens were skipped
            continue

        cur_page = int( token_r.attrib['page'] )

        if prev_page != cur_page:
            i_p += 1
            i_pt = 0
            prev_page = cur_page

        token_pl = page_level_tokens[i_p][i_pt]
        token_pl_text = token_pl[4][6:]

        if token_r.text != token_pl_text:
            print( "error! misaligned data!" )
            break

        label_cur = page_level_labels[i_p][i_pt]

        if label_cur == 'SCRAP':
            if cur_elm.tag == 'container' and cur_elm.attrib['name'] == 'dictScrap':
                cur_elm.append( token_r )
            else:
                scrap_cont = ET.SubElement( cur_elm, 'container', attrib={'name' : 'dictScrap'} )
                cur_elm = scrap_cont
                cur_elm.append( token_r )

        elif label_cur == 'ENTRY_START':
            entry_cont = ET.SubElement( body_elm, 'container', attrib={'name' : 'entry'} )
            cur_elm = entry_cont
            cur_elm.append( token_r )

        elif label_cur == 'ENTRY_INSIDE':
            if cur_elm.tag != 'container' or (cur_elm.tag == 'container' and cur_elm.attrib['name'] != 'entry'):
                entry_cont = ET.SubElement( body_elm, 'container', attrib={'name': 'entry'} )
                cur_elm = entry_cont

            # TODO implement other levels, ENTRY and SENSE levels here

            cur_elm.append( token_r )

        i_pt += 1


    xml_string = ET.tostring( doc_elm, encoding='unicode', method='xml' )
    with open( xml_out_file, 'w' ) as f:
        f.write( xml_string )

    return xml_string



if __name__ == "__main__":

    json_ml_results_file = '/media/jan/Fisk/CJVT/outputs/json/irish_trained.json'
    xml_raw_file = '/media/jan/Fisk/CJVT/data/dicts_xml_november/IrishSample_20p.xml'
    xml_out_file = '/media/jan/Fisk/CJVT/outputs/json/irish_out.xml'

    json2xml( json_ml_results_file, xml_raw_file, xml_out_file )


