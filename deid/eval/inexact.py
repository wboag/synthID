
import sys
import glob
import os
import nltk


base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tools_dir = os.path.join(base_path, 'tools')
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from tools import read_txt, read_tags





def main():

    try:
        txt_files  = sys.argv[1]
        pred_files = sys.argv[2]
        ref_files  = sys.argv[3]
    except Exception, e:
        print >>sys.stderr, '\n\tusage: python %s <txt-files> <pred-files> <ref-files> [--binary]\n'%sys.argv[0]
        exit(1)


    ######################################################################
    #                              READ DATA                             #
    ######################################################################

    txt_seen  = set()
    pred_seen = set()
    ref_seen  = set()


    for tags_file in glob.glob(txt_files):
        key = '.'.join(os.path.split(tags_file)[1].split('.')[:-1])
        txt_seen.add(key)

    for tags_file in glob.glob(pred_files):
        key = '.'.join(os.path.split(tags_file)[1].split('.')[:-1])
        pred_seen.add(key)

    for tags_file in glob.glob(ref_files):
        key = '.'.join(os.path.split(tags_file)[1].split('.')[:-1])
        ref_seen.add(key)

    seen = txt_seen & pred_seen & ref_seen

    if '--binary' in sys.argv:
        do_binary = True
    else:
        do_binary = False

    # read all text files for one large batch training set
    sentences = {}
    for txt_file in glob.glob(txt_files):
        key = '.'.join(os.path.split(txt_file)[1].split('.')[:-1])
        if key not in seen:
            continue
        s = read_txt(txt_file)
        sentences[key] = s

    # read references
    ref = {}
    categories = set()
    for tags_file in glob.glob(pred_files):
        key = '.'.join(os.path.split(tags_file)[1].split('.')[:-1])
        if key not in seen:
            continue

        s = sentences[key]

        t,c = read_tags(s, tags_file, do_binary=do_binary)

        ref[key] = t
        categories.update(c)

    # read predictions
    pred = {}
    for tags_file in glob.glob(ref_files):
        key = '.'.join(os.path.split(tags_file)[1].split('.')[:-1])
        if key not in seen:
            continue

        s = sentences[key]
        t,c = read_tags(s, tags_file, do_binary=do_binary)

        pred[key] = t
        categories.update(c)

    # this excludes an entire file if there is an alignment problem between sentences
    # (problem would come from tokenization issues)
    sents = []
    pred_tags = []
    ref_tags  = []
    for key in seen:
        if len(pred[key]) == len(ref[key]):
            sents      += sentences[key]
            pred_tags  += pred[key]
            ref_tags   += ref[key]


    ######################################################################
    #                            EVALUATION                              #
    ######################################################################

    # must have a prediction for every gold token
    assert len(pred_tags) == len(ref_tags)
    for i in range(len(ref_tags)):
        assert len(pred_tags[i]) == len(ref_tags[i])


    # token-level (approximately inexact span) f1
    categories.add('O')
    confusion = { ref:{ pred:0 for pred in categories} for ref in categories }
    for i in range(len(ref_tags)):
        for j in range(len(ref_tags[i])):
            gold = get_category( ref_tags[i][j])
            pred = get_category(pred_tags[i][j])

            confusion[gold][pred] += 1

    for c in categories:
        print '%-10s' % c, confusion[c]
    print '\n'


    # compute precision, recall, and f1
    weights    = []
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

        weights.append(all_actual)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print '\tprecision: %.3f (%5d/%5d)' % (precision,confusion[c][c],all_pred  )
        print '\trecall:    %.3f (%5d/%5d)' % (recall   ,confusion[c][c],all_actual)
        print '\tf1:        ', f1
        print

    # compute weighted average F1 score
    Z = sum(weights)
    weights = [ float(w)/Z for w in weights ]

    print 
    print '-'*40
    print
    print 'AVG'
    print '\tprecision: ', dot(precisions, weights)
    print '\trecall:    ', dot(recalls   , weights)
    print '\tf1:        ', dot(f1s       , weights)
    print 



def dot(u,v):
    val = 0
    for i in range(len(u)):
        val += u[i] * v[i]
    return val



def get_category(label):
    if label == 'O':
        return 'O'
    else:
        return label[2:]




if __name__ == '__main__':
    main()


