import xml.etree.ElementTree as ET
import json



def get_parent_container( e, parent_map ):
    # avoid self
    if e.tag.lower() == 'container':
        if e not in parent_map.keys():
            return None
        e = parent_map[e]
    # find container
    while True:
        if e.tag.lower() == 'container':
            return e
        if e not in parent_map.keys():
            return None
        e = parent_map[e]


def construct_containers_map( root, parent_map ):
    containers = list( root.iter( 'container' ) )
    containers_map = {}

    for cntnr in containers:
        # sometimes empty containers appear; skip those
        if len( list( cntnr.iter( 'TOKEN' ) ) ) == 0:
            continue

        parents_cur = []
        parent_cntnr = get_parent_container( cntnr, parent_map )
        while parent_cntnr is not None:
            parents_cur.append( parent_cntnr )
            parent_cntnr = get_parent_container( parent_cntnr, parent_map )
        containers_map[cntnr] = parents_cur

    return containers_map



def xml2json( xml_raw_file, xml_lex_file, json_out_file ):


    tree_raw = ET.parse( xml_raw_file )
    root_raw = tree_raw.getroot()
    tokens_raw = list( root_raw.iter( 'TOKEN' ) )

    tree_lex = ET.parse( xml_lex_file )
    parent_map = {c:p for p in tree_lex.iter() for c in p}
    root_lex = tree_lex.getroot()
    tokens_lex = list( root_lex.iter( 'TOKEN' ) )
    containers_map = construct_containers_map( root_lex, parent_map )
    container_names = []
    for c in root_lex.iter( 'container' ):
        if c.attrib['name'] not in container_names:
            container_names.append( c.attrib['name'].lower() )

    containers_levels = []
    for cntr in containers_map.keys():
        np = len( containers_map[cntr] )
        nm = cntr.attrib['name']
        while len( containers_levels ) < np + 1:
            containers_levels.append( [] )
        if nm not in containers_levels[np]:
            containers_levels[np].append( nm )

    data_lvl1 = []
    data_lvl2 = []
    data_lvl3 = []
    feats_lvl1 = []
    feats_lvl2 = []
    feats_lvl3 = []
    labels_lvl1 = []
    labels_lvl2 = []
    labels_lvl3 = []

    base_lvl1 = 'body'
    base_lvl2 = 'entry'
    base_lvl3 = 'sense'

    prev_lvl2_base = None
    prev_lvl3_base = None

    page_n = int( tokens_raw[0].attrib['page'] )
    line_n = 0
    # iterate over the annotated data that will be used for ML training
    for idx, token_a in enumerate( tokens_lex ):

        token_r = tokens_raw[idx]
        if not token_r.text == token_a.text:
            print( 'Misaligned data at token {}:{}!\nBreaking...'.format( idx, token_a ) )
            break

        if token_r.text is None:
            continue    # jump over empty tokens

        page_token = int( token_r.attrib['page'] )
        if page_token != page_n:
            data_lvl1.append( (feats_lvl1, labels_lvl1) )
            # data_lvl1.append(feats_pages)
            feats_lvl1 = []
            labels_lvl1 = []
            page_n = page_token

        # extract features
        feat = ['SIZE=' + token_r.attrib['font-size'], 'BOLD=' + token_r.attrib['bold'], 'ITALIC=' + token_r.attrib['italic'], 'FONT=' + token_r.attrib['font-name'], 'TOKEN=' + token_r.text]

        # newline feature
        line_token = int( token_r.attrib['line'] )
        if line_token != line_n:
            feat.append( 'NEWLINE' )
            line_n = line_token




        # # define containers
        # entry = None
        # form = None
        # pos = None
        # sense = None
        # # variant = None
        # trans = None

        # collect all the containers
        containers_cur = {}
        container = get_parent_container( token_a, parent_map )
        while container is not None:

            containers_cur[container.attrib['name']] = container

            # if container.attrib['name'] == 'entry':
            #     entry = container
            # elif container.attrib['name'] == 'form':
            #     form = container
            # elif container.attrib['name'] == 'pos':
            #     pos = container
            # elif container.attrib['name'] == 'sense':
            #     sense = container
            # # elif container.attrib['name'] == 'variant':
            # #     variant = container
            # elif container.attrib['name'] == 'translation':
            #     trans = container

            container = get_parent_container( container, parent_map )

        # # page level labels
        # if entry is not None:
        #
        #     if token_a == next( entry.iter('TOKEN') ):          # if the token_a is the first TOKEN in entry
        #         label_p = 'ENTRY_START'
        #     else:
        #         label_p = 'ENTRY_INSIDE'
        #
        #     # entry level labels
        #     if form is not None:
        #         label_e = 'FORM'
        #     elif pos is not None:
        #         label_e = 'POS'
        #     elif sense is not None:
        #
        #         if token_a == next( sense.iter('TOKEN') ):      # if the token_a is the first TOKEN in sense
        #             label_e = 'SENSE_START'
        #         else:
        #             label_e = 'SENSE_INSIDE'
        #
        #         # sense level labels
        #         if trans is not None:
        #             label_s = 'TRANS'
        #         else:
        #             label_s = 'INSIDE'
        #
        #         # write the complete sense and start a new one
        #         if sense != prev_sense and prev_sense is not None:
        #             data_lvl3.append( (feats_lvl3, labels_sense) )
        #             feats_lvl3 = []
        #             labels_sense = []
        #
        #         feats_lvl3.append( feat )
        #         labels_sense.append( label_s )
        #         prev_sense = sense
        #
        #     else:
        #         label_e = 'INSIDE'
        #
        #     # write the complete entry and start a new one
        #     if entry != prev_entry and prev_entry is not None:
        #         data_lvl2.append( (feats_entry, labels_entry) )
        #         feats_entry = []
        #         labels_entry = []
        #
        #     feats_entry.append( feat )
        #     labels_entry.append( label_e )
        #     prev_entry = entry
        #
        # else:
        #
        #     label_p = 'SCRAP'

        # get level of current token
        cur_level = len( containers_cur.keys() )
        if cur_level == 0 or 'scrap' in containers_cur.keys() or 'dictscrap' in containers_cur.keys():
            label_lvl1 = 'scrap'
        else:
            for cntr in containers_cur.keys():
                cur_lvl = 3










        feats_lvl1.append( feat )
        labels_lvl1.append( label_lvl1 )



    # add all the training data that is not yet added
    data_lvl1.append( (feats_lvl1, labels_lvl1) )
    data_lvl2.append( (feats_lvl2, labels_lvl2) )
    data_lvl3.append( (feats_lvl3, labels_lvl3) )


    # save training data into JSONs
    # deprecated
    # json.dump( data_lvl1, open( json_pages, 'w' ), indent=4 )
    # json.dump( data_lvl2, open( json_entries, 'w' ), indent=4 )
    # json.dump( data_lvl3, open( json_senses, 'w' ), indent=4 )




    # prepare the rest of the data that will go into ML for prediction
    index_start = 0
    page_n = int( tokens_raw[0].attrib['page'] )
    unlabelled_pages = []   # order by pages, such will be the 1st level input
    feats_lvl1 = []
    line_n = 0
    for token_p in tokens_raw:

        if token_p.text is None:        # sometimes empty tokens appear
            continue

        page_token = int( token_p.attrib['page'] )
        if page_token != page_n and len( feats_lvl1 ) != 0:
            unlabelled_pages.append( feats_lvl1 )
            feats_lvl1 = []
        feat = ['SIZE=' + token_p.attrib['font-size'], 'BOLD=' + token_p.attrib['bold'], 'ITALIC=' + token_p.attrib['italic'], 'FONT=' + token_p.attrib['font-name'], 'TOKEN=' + token_p.text]

        line_token = int( token_p.attrib['line'] )
        if line_token != line_n:
            feat.append( 'NEWLINE' )
            line_n = line_token

        feats_lvl1.append( feat )
        page_n = page_token


    unlabelled_pages.append( feats_lvl1 )
    # deprecated
    # json.dump( unlabelled_pages, open( json_unlabelled, 'w' ), indent=4 )

    json_dict = {'pages' : data_lvl1,
                 'entries' : data_lvl2,
                 'senses' : data_lvl3,
                 'unlabelled' : unlabelled_pages}

    json.dump( json_dict, open( json_out_file, 'w' ), indent=4 )
    return json_dict




if __name__ == "__main__":

    # inputs
    # xml_raw = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/mali_sloang_pred_prelomom-20-pages.xml'
    # xml_lex = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/mali_sloang_pred_prelomom-annotated.xml'
    xml_raw = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/sloita_proba5a-20-pages.xml'
    xml_lex = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/sloita_proba5a-annotated.xml'
    # xml_raw = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/srbslo_2_kor-20-pages.xml'
    # xml_lex = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/srbslo_2_kor-annotated.xml'

    # outputs
    # json_pages = '/media/jan/Fisk/CJVT/outputs/json/irish_pages.json'
    # json_entries = '/media/jan/Fisk/CJVT/outputs/json/irish_entries.json'
    # json_senses = '/media/jan/Fisk/CJVT/outputs/json/irish_senses.json'
    # json_unlabelled = '/media/jan/Fisk/CJVT/outputs/json/irish_unlabeled.json'
    json_out = '/media/jan/Fisk/CJVT/outputs/json/mali_sloang_packed_new.json'

    json_d = xml2json( xml_raw, xml_lex, json_out )



