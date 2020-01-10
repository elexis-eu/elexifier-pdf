import json
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Masking, Dense, Bidirectional, LSTM, TimeDistributed, Concatenate, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import argparse
from datetime import datetime
from time import time
import os





# parser=argparse.ArgumentParser()
# parser.add_argument('-p', '--percentage', help='percentage of the training data to be used', type=float, default=1.0)
# parser.add_argument('-v', '--verbose', action='store_true')
# parser.add_argument('-nr', '--n_rounds', help='number of repetitions of 10 epoch training', type=int, default=10)
# parser.add_argument('-ld', '--logdir', help='directory where the log file will be stored', type=str, default="")
# args = parser.parse_args()





def extract_tokens( dataset ):
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



def one_hot_and_chars( dataset, idx={}, idx_c={} ):
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



def one_hot_target( dataset, idx={} ):
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



def model_cLSTM( input_shape_feat, input_shape_char, output_len ):

    # model definition
    features_input = Input( shape=input_shape_feat )
    chars_input = Input( shape=input_shape_char )
    chars_masked = Masking( 0 ) (chars_input)
    features_masked = Masking( 0 ) (features_input)
    chars_embed = Bidirectional( LSTM( 8, return_sequences=True ) ) (chars_masked)
    chars_embed = Dropout( 0.4 ) (chars_embed)
    features_merged = Concatenate( axis=-1 ) ([features_masked, chars_embed])

    h = Bidirectional( LSTM( 20, return_sequences=True ) ) (features_merged)
    h = Dropout( 0.4 ) (h)
    # h = Bidirectional( LSTM( 8, return_sequences=True ) ) (h)
    # h = Dropout(0.25) (h)
    y = TimeDistributed( Dense( output_len, activation='softmax' ) ) (h)
    model = Model( inputs=[features_input, chars_input], outputs=y )
    print( model.summary() )

    return model



def train_on_data( data, n_rounds=10, verbose=True, logdir="logs/", batch_size=5 ):

    dt = datetime.now().strftime( "%Y%m%d-%H%M%S" )
    logfile = None
    if verbose and logdir is not "":
        if not os.path.exists( logdir ):
            os.makedirs( logdir )
        logfile = os.path.join( logdir, "train_c-lstm_{}.log".format(dt) )

    # data preparation
    # data = json.load( open( data_file, 'r' ) )
    n_train = int( round( len( data )*0.75 ) )
    train = data[:n_train]
    test = data[n_train:]

    x_train = [e[0] for e in train]
    y_train = [e[1] for e in train]
    x_test = [e[0] for e in test]
    y_test = [e[1] for e in test]

    all_tokens, idx_t, idx_c = extract_tokens( x_train )
    x_train_oh, x_train_chars, idx = one_hot_and_chars( x_train, idx_c=idx_c )
    x_test_oh, x_test_chars, _ = one_hot_and_chars( x_test, idx, idx_c )

    max_sequence_len = max( [len(e) for e in x_train_oh] )*2
    x_train_oh = sequence.pad_sequences( x_train_oh, max_sequence_len )
    x_test_oh = sequence.pad_sequences( x_test_oh, max_sequence_len )

    max_carray_len = 0
    for seq in x_train_chars:
        max_cur = max( [len(arr) for arr in seq] )
        if max_cur > max_carray_len:
            max_carray_len = max_cur
    max_carray_len *= 2

    x_train_chars_pad = [sequence.pad_sequences( seq, max_carray_len ) for seq in x_train_chars]
    x_train_chars = sequence.pad_sequences( x_train_chars_pad, max_sequence_len )
    x_test_chars_pad = [sequence.pad_sequences( seq, max_carray_len ) for seq in x_test_chars]
    x_test_chars = sequence.pad_sequences( x_test_chars_pad, max_sequence_len )
    X_train = [x_train_oh, x_train_chars]
    X_test = [x_test_oh, x_test_chars]

    y_train_oh, idx_label = one_hot_target( y_train )
    y_test_oh, _ = one_hot_target( y_test, idx_label )
    y_train_oh = sequence.pad_sequences( y_train_oh, max_sequence_len )
    y_test_oh = sequence.pad_sequences( y_test_oh, max_sequence_len )
    rev_idx_label = {v:k for k, v in idx_label.items()}

    data_infos = {'idx': idx,
                  'idx_c': idx_c,
                  'idx_label': idx_label,
                  'max_sequence_len': max_sequence_len,
                  'max_carray_len': max_carray_len}

    model = model_cLSTM( (max_sequence_len, len( idx )), (max_sequence_len, max_carray_len), len( idx_label ) )
#    optim = Adam( lr=0.005 )
#    optim = SGD( lr=0.01, momentum=0.9, nesterov=True )
    optim = RMSprop( lr=0.005 )
    model.compile( loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'] )
    best_acc = 0
    best_model_path = 'best_models/best_model_' + dt + '.h5'
    # best_model_path = '/media/jan/Fisk/CJVT/models/pipeline_debug/best_model_' + dt + '.h5'
    # DEBUG #
    # best_model_path = "/media/jan/Fisk/CJVT/models/pipeline_debug/best_model_20191230-121243.h5"
    # model.load_weights( best_model_path )

    t1 = time()
    for i_round in range(n_rounds):

        if verbose: print('ROUND', i_round)
        t_r0 = time()

        h = model.fit( X_train, y_train_oh, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test_oh), shuffle=True )
        score, acc = model.evaluate( X_test, y_test_oh, batch_size=5 )
        if acc > best_acc:
            print( "best model accuracy,", acc, ", saving..." )
            best_acc = acc
            model.save_weights( best_model_path )

        if verbose:
            y_test_pred = model.predict( X_test )

            def to_tags(sequence):
                result=[]
                for label in sequence:
                    result.append(rev_idx_label[np.argmax(label)])
                return result

            lf = None
            if logfile is not None:
                lf = open(logfile, "a+")
                lf.write("ROUND {}\n\n".format(i_round))

            pred=[]
            true=[]
            from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
            for p,t,f in zip(y_test_pred, y_test_oh, x_test):
                for i,u in enumerate(t):
                    if sum(u)>0:
                        break
                if verbose:
                    for a,b,c in zip(to_tags(p[i:]),to_tags(t[i:]),f):
                        if a!=b:
                            # print('!',a,b,'|'.join(c))
                            if lf: lf.write('! {} {} {}\n'.format(a, b, '|'.join(c)))
                        else:
                            # print(a,b,'|'.join(c))
                            if lf: lf.write('{} {} {}\n'.format(a, b, '|'.join(c)))
                    # print("")
                    if lf: lf.write("\n")
                pred.extend(to_tags(p[i:]))
                true.extend(to_tags(t[i:]))

            print(confusion_matrix(true,pred))
            print(classification_report(true,pred))
            print("Accuracy:", accuracy_score(true,pred))
            print("ROUND time:", (time()-t_r0), "s")
            print("")

            if lf:
                lf.write("{}\n".format(confusion_matrix(true,pred)))
                lf.write("{}\n".format(classification_report(true,pred)))
                lf.write("Accuracy: {}\n".format(accuracy_score(true,pred)))
                lf.write("ROUND time: {} s\n".format((time()-t_r0)))
                lf.write("\n")
                lf.close()

    # in the end load the model with the best score
    model.load_weights( best_model_path )
    if verbose: print("Training time:", (time()-t1), "s")
    return model, data_infos



def train_ML( data_packed_file, json_out_file, logdir ):

    # train all 3 models
    data = json.load( open( data_packed_file, 'r' ) )
    model_pages, pages_infos = train_on_data( data['pages'], n_rounds=8, verbose=True, logdir=logdir, batch_size=4 )
    # DEBUG #
    # model_pages.load_weights( "/media/jan/Fisk/CJVT/models/pipeline_debug/best_model_20191230-121243.h5" )


    # predict on the rest of the data

    # 1.) pages level prediction
    pages_tokens = data['unlabelled']
    x_new = pages_tokens
    x_new_oh, x_new_chars, _ = one_hot_and_chars( x_new, pages_infos['idx'], pages_infos['idx_c'] )
    x_new_oh = sequence.pad_sequences( x_new_oh, pages_infos['max_sequence_len'] )
    x_new_chars_pad = [sequence.pad_sequences( seq, pages_infos['max_carray_len'] ) for seq in x_new_chars]
    x_new_chars = sequence.pad_sequences( x_new_chars_pad, pages_infos['max_sequence_len'] )
    X_new = [x_new_oh, x_new_chars]
    rev_idx_label_pages = {v: k for k, v in pages_infos['idx_label'].items()}

    y_pred_pages = model_pages.predict( X_new )

    pages_labels = []
    entries_tokens = []
    entry = []
    for i_p in range( len( pages_tokens ) ):

        n_tokens_in = len( pages_tokens[i_p] )

        page_labels = []
        for i_t in range( n_tokens_in ):

            # take care of correct indices with regard to padding and max_sequence_len
            if n_tokens_in <= pages_infos['max_sequence_len']:
                i_t_pad = i_t + pages_infos['max_sequence_len'] - n_tokens_in
            else:
                i_t_pad = i_t

            token_cur = pages_tokens[i_p][i_t]
            label_cur = rev_idx_label_pages[np.argmax( y_pred_pages[i_p][i_t_pad] )]
            page_labels.append( label_cur )

            if label_cur == 'SCRAP':
                continue

            if label_cur == 'ENTRY_START':

                if len( entry ) != 0:
                    entries_tokens.append( entry )
                entry = []

            entry.append( token_cur )

        pages_labels.append( page_labels )

    if len( entry ) != 0:
        entries_tokens.append( entry )



    # 2.) entries level prediction
    model_entries, entries_infos = train_on_data( data['entries'], n_rounds=8, verbose=True, logdir=logdir, batch_size=8 )
    # DEBUG #
    # model_entries.load_weights( "/home/jan/PycharmProjects/cjvt-dev/elexifier-pdf/tri_level_scripts/best_model_20191227-153452.h5" )
    x_new = entries_tokens
    x_new_oh, x_new_chars, _ = one_hot_and_chars( x_new, entries_infos['idx'], entries_infos['idx_c'] )
    x_new_oh = sequence.pad_sequences( x_new_oh, entries_infos['max_sequence_len'] )
    x_new_chars_pad = [sequence.pad_sequences( seq, entries_infos['max_carray_len'] ) for seq in x_new_chars]
    x_new_chars = sequence.pad_sequences( x_new_chars_pad, entries_infos['max_sequence_len'] )
    X_new = [x_new_oh, x_new_chars]
    rev_idx_label_entries = {v: k for k, v in entries_infos['idx_label'].items()}

    y_pred_entries = model_entries.predict( X_new )

    entries_labels = []
    senses_tokens = []
    sense = []
    for i_e in range( len( y_pred_entries ) ):

        n_tokens_in = len( entries_tokens[i_e] )
        entry_labels = []
        for i_t in range( n_tokens_in ):

            # take care of correct indices with regard to padding and max_sequence_len
            if n_tokens_in <= entries_infos['max_sequence_len']:
                i_t_pad = i_t + entries_infos['max_sequence_len'] - n_tokens_in
            else:
                i_t_pad = i_t

            token_cur = entries_tokens[i_e][i_t]
            if i_t_pad < entries_infos['max_sequence_len']:
                label_cur = rev_idx_label_entries[np.argmax( y_pred_entries[i_e][i_t_pad] )]
            else:
                label_cur = 'INSIDE'

            entry_labels.append( label_cur )

            if 'SENSE' not in label_cur:
                continue

            if label_cur == 'SENSE_START':

                if len( sense ) != 0:
                    senses_tokens.append( sense )
                sense = []

            sense.append( token_cur )

        entries_labels.append( entry_labels )

    if len( sense ) != 0:
        senses_tokens.append( sense )



    # 3.) senses level prediction
    model_senses, senses_infos = train_on_data( data['senses'], n_rounds=8, verbose=True, logdir=logdir, batch_size=8 )
    x_new = senses_tokens
    x_new_oh, x_new_chars, _ = one_hot_and_chars( x_new, senses_infos['idx'], senses_infos['idx_c'] )
    x_new_oh = sequence.pad_sequences( x_new_oh, senses_infos['max_sequence_len'] )
    x_new_chars_pad = [sequence.pad_sequences( seq, senses_infos['max_carray_len'] ) for seq in x_new_chars]
    x_new_chars = sequence.pad_sequences( x_new_chars_pad, senses_infos['max_sequence_len'] )
    X_new = [x_new_oh, x_new_chars]
    rev_idx_label_senses = {v: k for k, v in senses_infos['idx_label'].items()}

    y_pred_senses = model_senses.predict( X_new )

    senses_labels = []
    for i_s in range( len( senses_tokens ) ):

        n_tokens_in = len( senses_tokens[i_s] )
        sense_labels = []
        for i_t in range( n_tokens_in ):

            # take care of correct indices with regard to padding and max_sequence_len
            if n_tokens_in <= senses_infos['max_sequence_len']:
                i_t_pad = i_t + senses_infos['max_sequence_len'] - n_tokens_in
            else:
                i_t_pad = i_t

            # token_cur = senses_tokens[i_s][i_t]
            if i_t_pad < senses_infos['max_sequence_len']:
                label_cur = rev_idx_label_senses[np.argmax( y_pred_senses[i_s][i_t_pad] )]
            else:
                label_cur = 'INSIDE'
            sense_labels.append( label_cur )

        senses_labels.append( sense_labels )

    json_data = {'page_level': (pages_tokens, pages_labels) ,
                 'entry_level': (entries_tokens, entries_labels),
                 'sense_level': (senses_tokens, senses_labels)
                }

    json.dump( json_data, open( json_out_file, 'w' ), indent=4 )

    return json_data




if __name__ == "__main__":

    # json_in_file = '/media/jan/Fisk/CJVT/outputs/json/mali_sloang_packed.json'
    json_in_file = '/home/jjug/data/slovarji/mali_sloang_packed.json'

    # json_out_file = '/media/jan/Fisk/CJVT/outputs/json/mali_sloang_trained.json'
    json_out_file = '/home/jjug/data/slovarji/mali_sloang_trained_5.json'

    logdir = "/home/jjug/logs/train_20200110"
    # logdir = ""

    jdata = train_ML( json_in_file, json_out_file, logdir )



