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



def get_container_structure( container, level=0, structure=[] ):
    if len( structure ) <= level:
        structure.append( [] )
    if container.attrib['name'] not in structure[level]:
        structure[level].append( container.attrib['name'] )
    for child in container.getchildren():
        if child.tag != 'container':
            continue
        structure = get_container_structure( child, level+1, structure )
    return structure





def get_all_container_structures( xml_lex_file ):
    tree_lex = ET.parse( xml_lex_file )
    parent_map = {c:p for p in tree_lex.iter() for c in p}
    root_lex = tree_lex.getroot()
    containers_map = construct_containers_map( root_lex, parent_map )

    # get all 1st level containers
    containers_lvl1 = []
    for cntr in containers_map.keys():
        if len( containers_map[cntr] ) == 0:
            containers_lvl1.append( cntr )

    structures = []
    struct_counts = []
    for cntr in containers_lvl1:
        structure = get_container_structure( cntr, 0, [] )
        if structure not in structures:
            structures.append( structure )
            struct_counts.append( 1 )
        else:
            for i in range( len( structures ) ):
                if structures[i] == structure:
                    struct_counts[i] += 1
                    break

    return structures, struct_counts



def add_senses_where_missing( tree_lex, entry_level_structure ):
    # this function is not the best, because it can change the order of TOKENs in some circumstances, which is not
    # desirable. See note below.
    # This function is deprecated as of 2020 01 24

    parent_map = {c: p for p in tree_lex.iter() for c in p}
    root_lex = tree_lex.getroot()
    containers_map = construct_containers_map( root_lex, parent_map )

    # get all entry containers
    entry_containers = []
    for cntr in containers_map.keys():
        if len( containers_map[cntr] ) == 0 and cntr.attrib['name'].lower() == 'entry':
            entry_containers.append( cntr )

    for entry in entry_containers:
        child_containers = []
        children_to_move = []
        indices_to_move = []
        children = entry.getchildren()
        last_entry_level_cntr_idx = 0
        for idx, child in enumerate( children ):
            if child.tag.lower() == 'token':
                children_to_move.append( child )
                indices_to_move.append( idx )
            elif child.tag.lower() == 'container':
                if child.attrib['name'] not in entry_level_structure:
                    children_to_move.append( child )
                    indices_to_move.append( idx )
                else:
                    last_entry_level_cntr_idx = idx

            if child.tag.lower() == 'container':
                child_containers.append( child )

        child_cont_names = [ch.attrib['name'] for ch in child_containers]
        if 'sense' in child_cont_names or len( children_to_move ) == 0:
            continue

        # NOTE: the sense container is always appended to the end of current entry, which means the order of tokens
        # in this tree and the raw file may not be the same anymore...
        # sense_container = ET.SubElement( entry, 'container', attrib={'name': 'sense'} )
        # for child in children_to_move:
        #     entry.remove( child )
        # for child in children_to_move:
        #     sense_container.append( child )

        # TEMP SOLUTION: move into sense only the TOKENs and sense-level containers that come after the last entry-level
        # container
        sense_container = ET.SubElement( entry, 'container', attrib={'name': 'sense'} )
        for ich, child in enumerate( children_to_move ):
            if indices_to_move[ich] > last_entry_level_cntr_idx:
                entry.remove( child )
        for ich, child in enumerate( children_to_move ):
            if indices_to_move[ich] > last_entry_level_cntr_idx:
                sense_container.append( child )

    return tree_lex



def xml2json( xml_raw_file, xml_lex_file, approved_structures, json_out_file ):

    tree_raw = ET.parse( xml_raw_file )
    root_raw = tree_raw.getroot()
    tokens_raw = list( root_raw.iter( 'TOKEN' ) )

    tree_lex = ET.parse( xml_lex_file )
    # tree_lex = add_senses_where_missing( tree_lex, container_structure[1] )

    root_lex = tree_lex.getroot()
    tokens_lex = list( root_lex.iter( 'TOKEN' ) )
    parent_map = {c: p for p in tree_lex.iter() for c in p}
    containers_map = construct_containers_map( root_lex, parent_map )

    # prepare variables and
    data_lvl1 = []
    data_lvl2 = []
    data_lvl3 = []
    feats_lvl1 = []
    feats_lvl2 = []
    feats_lvl3 = []
    labels_lvl1 = []
    labels_lvl2 = []
    labels_lvl3 = []

    prev_entry = None
    prev_sense = None

    entries_blacklist = []  # blacklist for entries whose structure does not conform to specifications

    page_n = int( tokens_raw[0].attrib['page'] )
    line_n = 0
    # iterate over the annotated data that will be used for ML training
    for idx, token_a in enumerate( tokens_lex ):

        token_r = tokens_raw[idx]
        if not token_r.text == token_a.text:
            print( 'Misaligned data at token {}:{}!\nBreaking...'.format( idx, token_a.text ) )
            break

        if token_r.text is None:
            continue    # jump over empty tokens

        page_token = int( token_r.attrib['page'] )
        if page_token != page_n:
            data_lvl1.append( (feats_lvl1, labels_lvl1) )
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


        # collect all the containers of current token
        containers_cur = []
        container = get_parent_container( token_a, parent_map )
        while container is not None:
            containers_cur.append( container )
            container = get_parent_container( container, parent_map )

        # find the base (lowest level) entry container
        entry_cur = None
        for cntr in containers_cur:
            if cntr.attrib['name'] == 'entry' and len( containers_map[cntr] ) == 0:
                entry_cur = cntr
                break

        # check if current entry was already identified as faulty
        if entry_cur in entries_blacklist:
            continue

        # check if current entry is faulty
        structure_cur = get_container_structure( entry_cur )
        if structure_cur not in approved_structures:
            entries_blacklist.append( entry_cur )
            continue

        # # perform check if container structure is correct:
        # containers_by_lvl = [None, None, None]
        # structure_correct = True
        # for cntr in containers_cur:
        #     cntr_lvl = len( containers_map[cntr] )
        #     if cntr_lvl >= len( container_structure ) or cntr.attrib['name'] not in container_structure[cntr_lvl]:
        #         structure_correct = False
        #         break
        #     else:
        #         containers_by_lvl[cntr_lvl] = cntr
        #
        # # if there was a problem with the structure, skip current token for training data
        # if not structure_correct:
        #     continue

        # identify containers of current token by level
        containers_by_lvl = [None, None, None]
        for cntr in containers_cur:
            cntr_lvl = len( containers_map[cntr] )
            containers_by_lvl[cntr_lvl] = cntr

        # page level labels
        if containers_by_lvl[0] is not None:

            if containers_by_lvl[0].attrib['name'].lower() == 'entry':
                entry = containers_by_lvl[0]
                if token_a == next( entry.iter('TOKEN') ):          # if the token_a is the first TOKEN in entry
                    label_lvl1 = 'entry_start'
                else:
                    label_lvl1 = 'entry_inside'

                # entry level labels
                if containers_by_lvl[1] is not None:

                    if containers_by_lvl[1].attrib['name'].lower() == 'sense':
                        sense = containers_by_lvl[1]
                        if token_a == next( sense.iter( 'TOKEN' ) ):  # if the token_a is the first TOKEN in sense
                            label_lvl2 = 'sense_start'
                        else:
                            label_lvl2 = 'sense_inside'

                        # sense level labels
                        if containers_by_lvl[2] is not None:
                            label_lvl3 = containers_by_lvl[2].attrib['name']
                        else:  # if token has no container beneath sense
                            label_lvl3 = '0'

                        # if this is new sense, write the previous sense and start a new one
                        if sense != prev_sense and prev_sense is not None:
                            data_lvl3.append( (feats_lvl3, labels_lvl3) )
                            feats_lvl3 = []
                            labels_lvl3 = []

                        feats_lvl3.append( feat )
                        labels_lvl3.append( label_lvl3 )
                        prev_sense = sense

                    else:   # if level 2 container is not an entry
                        label_lvl2 = containers_by_lvl[1].attrib['name']

                else:   # if token has no container beneath entry
                    label_lvl2 = '0'

                # if this is new entry, write the previous entry and start a new one
                if entry != prev_entry and prev_entry is not None:
                    data_lvl2.append( (feats_lvl2, labels_lvl2) )
                    feats_lvl2 = []
                    labels_lvl2 = []

                feats_lvl2.append( feat )
                labels_lvl2.append( label_lvl2 )
                prev_entry = entry

            else:   # if level 1 container is not an entry (by some possible structures)
                label_lvl1 = containers_by_lvl[0].attrib['name']

        else:   # if token has no container

            label_lvl1 = 'scrap'

        feats_lvl1.append( feat )
        labels_lvl1.append( label_lvl1 )



    # add all the training data that is not yet added
    data_lvl1.append( (feats_lvl1, labels_lvl1) )
    data_lvl2.append( (feats_lvl2, labels_lvl2) )
    data_lvl3.append( (feats_lvl3, labels_lvl3) )


    # prepare the rest of the data that will go into ML for prediction
    index_start = 0         # this index determines where data for ML prediction starts. If 0, even the already annotated data from lexonomy will be predicted
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
    json_dict = {'level_1' : data_lvl1,
                 'level_2' : data_lvl2,
                 'level_3' : data_lvl3,
                 'unlabelled' : unlabelled_pages}

    json.dump( json_dict, open( json_out_file, 'w' ), indent=4 )
    return json_dict




if __name__ == "__main__":

    # inputs
    xml_raw = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/mali_sloang_pred_prelomom-20-pages.xml'
    xml_lex = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/mali_sloang_pred_prelomom-annotated.xml'
    # xml_raw = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/sloita_proba5a-20-pages.xml'
    # xml_lex = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/sloita_proba5a-annotated.xml'
    # xml_raw = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/srbslo_2_kor-20-pages.xml'
    # xml_lex = '/media/jan/Fisk/CJVT/data/dicts_xml_december/slovarji/srbslo_2_kor-annotated.xml'

    # structures, count  = get_all_container_structures( xml_lex )
    # container_structure = [['entry'], ['form', 'pos', 'sense'], ['translation']]
    container_structure = [['entry'], ['form', 'pos', 'variant', 'sense'], ['translation']]

    # outputs
    # json_pages = '/media/jan/Fisk/CJVT/outputs/json/irish_pages.json'
    # json_entries = '/media/jan/Fisk/CJVT/outputs/json/irish_entries.json'
    # json_senses = '/media/jan/Fisk/CJVT/outputs/json/irish_senses.json'
    # json_unlabelled = '/media/jan/Fisk/CJVT/outputs/json/irish_unlabeled.json'
    json_out = '/media/jan/Fisk/CJVT/outputs/json/mali_sloang_packed_final.json'
    # json_out = '/media/jan/Fisk/CJVT/outputs/json/sloita_proba5a_packed.json'
    # json_out = '/media/jan/Fisk/CJVT/outputs/json/srbslo_2_kor_packed.json'

    json_d = xml2json( xml_raw, xml_lex, container_structure, json_out )



