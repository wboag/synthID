
import sys
import nltk
import cPickle as pickle
import glob
import os

import crf


base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tools_dir = os.path.join(base_path, 'tools')
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from tools import flatten, save_list_structure, reconstruct_list
from tools import read_txt, read_tags, extract_features


from sklearn.feature_extraction import DictVectorizer




def main():

    try:
        txt_files   = sys.argv[1]
        tags_files  = sys.argv[2]
        model_path  = sys.argv[3]
    except Exception, e:
        print >>sys.stderr, '\n\tusage: python %s [text-file] [tags-file] [model-path]\n'%sys.argv[0]
        exit(1)


    ######################################################################
    #                              READ DATA                             #
    ######################################################################

    txt_file_list = glob.glob(txt_files)

    if '--limit' in sys.argv:
        N = int(sys.argv[sys.argv.index('--limit')+1])
    else:
        N = len(txt_file_list)

    txt_file_list = txt_file_list[:N]

    # read all text files for one large batch training set
    sentences = {}
    for txt_file in txt_file_list:
        key = '.'.join(os.path.split(txt_file)[1].split('.')[:-1])
        s = read_txt(txt_file)
        sentences[key] = s

    print 'training on %d files' % N

    # flag that allows training on binary PHI (rather than specifically name/number/etc)
    if '--binary' in sys.argv:
        do_binary = True
    else:
        do_binary = False

    # read tags
    sents = []
    tags = []
    categories = set()
    for tags_file in glob.glob(tags_files):
        key = '.'.join(os.path.split(tags_file)[1].split('.')[:-1])
        if key not in sentences:
            continue

        s = sentences[key]

        t,c = read_tags(s, tags_file, do_binary=do_binary)

        sents += s
        tags  += t
        categories.update(c)


    ######################################################################
    #                            FEATURE ENGINEERING                     #
    ######################################################################

    text_features = extract_features(sents)

    ######################################################################
    #                         FORMATTING DATA                            #
    ######################################################################

    # text features -> sparse numeric design matrix
    dvec = DictVectorizer()

    # convert text features to design matrix
    offsets = save_list_structure(text_features)
    flat_text_features = flatten(text_features)
    flat_train_X = dvec.fit_transform(flat_text_features)

    # vectorize labels
    flat_tags = flatten(tags)

    tag2ind = { tag:i for i,tag in enumerate(set(flat_tags)) }
    ind2tag = { i:tag for tag,i in tag2ind.items()           }
    
    flat_train_Y = [ tag2ind[tag] for tag in flat_tags ]
    
    # reconstruct list structures
    train_X = reconstruct_list( list(flat_train_X) , offsets)
    train_Y = reconstruct_list(      flat_train_Y  , offsets)

    # build CRF model
    crf_model = crf.train(train_X, train_Y)

    # save all important info
    with open(model_path, 'wb') as f:
        pickle.dump(dvec     , f)
        pickle.dump(tag2ind  , f)
        pickle.dump(crf_model, f)


    ######################################################################
    #                            PREDICTING                              #
    ######################################################################


    # test it out
    pred_Y = crf.predict(crf_model, train_X)
    pred_tags = [ [ind2tag[p] for p in P] for P in pred_Y ]


    ######################################################################
    #                            EVALUATION                              #
    ######################################################################


    # token-level (approximately inexact span) f1
    categories.add('O')
    confusion = { ref:{ pred:0 for pred in categories} for ref in categories }
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            gold = get_category(     tags[i][j])
            pred = get_category(pred_tags[i][j])

            confusion[gold][pred] += 1


    for c in categories:
        print '%-10s' % c, confusion[c]
    print '\n'


    # compute precision, recall, and f1
    precisions = []
    recalls    = []
    f1s        = []
    for c in categories:
        if c == 'O': continue

        print c

        all_pred   = sum([confusion[c][p] for p in categories])
        all_actual = sum([confusion[g][c] for g in categories])

        # precision
        precision = confusion[c][c] / (1e-9 + all_pred)
        recall    = confusion[c][c] / (1e-9 + all_actual)

        f1 = (2*precision*recall) / (1e-9 + precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print '\tprecision: %.3f (%5d/%5d)' % (precision,confusion[c][c],all_pred  )
        print '\trecall:    %.3f (%5d/%5d)' % (recall   ,confusion[c][c],all_actual)
        print '\tf1:        ', f1
        print


    print 
    print '-'*40
    print
    print 'AVG'
    print '\tprecision: ', sum(precisions) / len(precisions)
    print '\trecall:    ', sum(recalls) / len(recalls)
    print '\tf1:        ', sum(f1s) / len(f1s)
    print 





def get_category(label):
    if label == 'O':
        return 'O'
    else:
        return label[2:]



if __name__ == '__main__':
    main()


