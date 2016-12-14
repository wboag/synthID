######################################################################
#  CliNER - tools.py                                                 #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: General purpose tools                                    #
######################################################################


import nltk
import os
import re



def flatten(list_of_lists):

    '''
    flatten()

    Purpose: Given a list of lists, flatten one level deep

    @param list_of_lists. <list-of-lists> of objects.
    @return               <list> of objects (AKA flattened one level)

    >>> flatten([['a','b','c'],['d','e'],['f','g','h']])
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    '''

    return [item for sublist in list_of_lists for item in sublist]




def save_list_structure(list_of_lists):

    '''
    save_list_structure()

    Purpose: Given a list of lists, save way to recover structure from flattended

    @param list_of_lists. <list-of-lists> of objects.
    @return               <list> of indices, where each index refers to the
                                 beginning of a line in the orig list-of-lists

    >>> save_list_structure([['a','b','c'],['d','e'],['f','g','h']])
    [3, 5, 8]
    '''

    offsets = [ len(sublist) for sublist in list_of_lists ]
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i-1]

    return offsets




def reconstruct_list(flat_list, offsets):

    '''
    save_list_structure()

    Purpose: This undoes a list flattening. Uses value from save_list_structure()

    @param flat_list. <list> of objects
    @param offsets    <list> of indices, where each index refers to the
                                 beginning of a line in the orig list-of-lists
    @return           <list-of-lists> of objects (the original structure)

    >>> reconstruct_list(['a','b','c','d','e','f','g','h'], [3,5,8])
    [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h']]
    '''

    return [ flat_list[i:j] for i, j in zip([0] + offsets, offsets)]




def read_txt(txt_file):
    # read text
    with open(txt_file, 'r') as f:
        text = f.read()
        sents = [ nltk.word_tokenize(line.strip()) for line in text.split('\n') ]
    return sents

    # TODO: ACTUALLY WORK TO MAKE COHERENT SENTENCES

    text = re.sub('\n\n+', '\n\n', text)

    segments = text.split('\n\n')

    # Is this segment nonprose?
    sents = []
    for segment in segments:
        print '-'*50
        #print segment
        segment_sents = nltk.sent_tokenize(segment)
        for sent in segment_sents:
            print '='*30
            print sent
            lines = sent.split('\n')
            buf = []
            for line in lines:
                if line.endswith(':'):
                    sents.append(line)
                    print '\t>>', line
                elif re.search('^\d+\.$', line):
                    sents.append(line)
                    print '\t##', line
                else:
                    if buf and line[0].isupper():
                        final = ' '.join(buf)
                        print '\t--', final
                        sents.append(final)
                        buf = []
                    buf.append(line)

            if buf:
                final = ' '.join(buf)
                print '\t--', final
                sents.append(final)
            print '='*30
            print 
        print '-'*50
        print 


    '''
    print '\n\n\n'
    print text
    print '\n\n\n'
    print segments
    print '\n\n\n'
    '''
    exit()

    return sents



def read_tags(sents, tags_file, do_binary=False):
    # read tags
    annotations = []
    with open(tags_file, 'r') as f:
        for a_line in f.readlines():
            toks = a_line.strip().split('\t')
            assert int(toks[0].split(':')[0])  == int(toks[1].split(':')[0])
            lineno = int(toks[0].split(':')[0])
            start  = int(toks[0].split(':')[1])
            end    = int(toks[1].split(':')[1])
            segment = toks[2][7:-2]
            label   = toks[3][8:-2]
            if do_binary:
                label = 'PHI'
            annotations.append( (lineno,start,end,segment,label) )

    # construct sequence tags
    tags = [ ['O' for word in sent] for sent in sents ]
    categories = set()
    for lineno,start,end,segment,label in annotations:
        '''
        print 
        print sents[lineno][start:end+1]
        print nltk.word_tokenize(segment)
        '''

        # TEMPORARY HACK
        if sents[lineno][start:end+1] != nltk.word_tokenize(segment):
            continue

        assert sents[lineno][start:end+1] == nltk.word_tokenize(segment)
        #print 'WARNING: skipping assertions of len(tokens) == len(tags)'

        tags[lineno][start] = 'B-%s' % label
        for i in range(end-start):
            tags[lineno][start+i+1] = 'I-%s' % label

        categories.add(label)

    return tags, categories



def print_predictions(tags_file, sents, pred_tags):

    # output to tags format
    with open(tags_file, 'w') as f:
        for lineno,preds in enumerate(pred_tags):
            spans = []
            active = False
            start = None
            for i in range(len(preds)):
                if preds[i][0] == 'B':
                    # if there's a previous adjacent span (BIB or BB), then end that one
                    if active:
                        spans.append( (start,i-1) )
                    # begin new span
                    start = i
                    active = True
                elif preds[i][0] == 'O':
                    if active:
                        spans.append( (start,i-1) )
                    active = False

            # if something left, flush it
            if active:
                spans.append( (start,len(preds)-1) )

            for start,end in spans:
                entity = ' '.join(sents[lineno][start:end+1])
                label  = pred_tags[lineno][start][2:]
                print >>f, '%d:%d\t%d:%d\ttext=[[%s]]\tlabel=[[%s]]' % (lineno,start,lineno,end,entity,label)




def extract_features(sents):

    # TODO
    #   1. try a padded context thing (TRYING THIS)
    #   2. identify prose sections & run sentence tokenizer on that (TOO HARD)

    copies = 10

    # makes it easier to gather previous context
    prev_N = 3
    all_words = ['<PAD>' for _ in range(prev_N)]
    lineno_to_span = {}
    for i,sent in enumerate(sents):
        start = len(all_words)
        if sent == []:
            sent = ['<BLANK>']
        all_words += sent
        end = len(all_words)-1
        lineno_to_span[i] = (start,end)
    all_words += ['<PAD>' for _ in range(prev_N)]
    all_words = [ w.lower() for w in all_words ]

    V = set(all_words)
    print 'vocab: ', len(V)
    print '\n\n'

    #print all_words
    #exit()

    # get some features for each word of each sentence
    text_features = []
    for lineno,sent in enumerate(sents):
        features_list = []

        start,end = lineno_to_span[lineno]

        for i,w in enumerate(sent):
            features = {'dummy':1}

            #'''
            # previous words (especially helpful for beginning-of-sentence words
            for j in range(1,prev_N+1):
                for k in range(copies):
                    prev_word = all_words[start+i-j]
                    features[('prev-unigram-%d'%j,k,prev_word)] = 1
            #'''

            # unigram (note: crfsuite has weird issues when given too few feats)
            for j in range(copies):
                features[('unigram',j,w.lower())] = 1

            # is this word a common name?
            for j in range(copies):
                if w.lower() in male_names:
                    features[('male_name',j)] = 1
                if w.lower() in female_names:
                    features[('female_name',j)] = 1
                if w.lower() in last_names:
                    features[('last_name',j)] = 1
                if w.lower() in hospitals:
                    features[('hospital',j)] = 1

            # next words (especially helpful for end-of-sentence words
            for j in range(1,prev_N+1):
                for k in range(copies):
                    prev_word = all_words[start+i+j]
                    features[('next-unigram-%d'%j,k,prev_word)] = 1

            '''
            print 
            print 'w: ', w
            print 'i: ', i
            print 'sent: ', sent
            print 'span: ', all_words[start:end+1]
            print features
            print 
            '''

            features_list.append(features)

        #print '\n\n'

        text_features.append(features_list)

    return text_features



def read_names(filename):
    names  = set()
    with open(filename, 'r') as f:
        for line in f.readlines():
            toks = line.strip().split()
            names.add(toks[0].lower())

    return names



# List of common names and hospials
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lexica_dir = os.path.join(parent_dir, 'synth', 'lexica', 'final')
male_names   = read_names(os.path.join(lexica_dir, 'names-male.txt'   ))
female_names = read_names(os.path.join(lexica_dir, 'names-female.txt' ))
last_names   = read_names(os.path.join(lexica_dir, 'names-surname.txt'))
hospitals    = read_names(os.path.join(lexica_dir, 'hospitals.txt'    ))


