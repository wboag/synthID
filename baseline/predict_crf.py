
import sys
import nltk
import cPickle as pickle
import glob
import os

import crf


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tools_dir = os.path.join(base_path, 'tools')
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from tools import flatten, save_list_structure, reconstruct_list
from tools import read_txt, print_predictions, extract_features


from sklearn.feature_extraction import DictVectorizer




def main():

    try:
        txt_files  = sys.argv[1]
        out_dir    = sys.argv[2]
        model_path = sys.argv[3]
    except Exception, e:
        print >>sys.stderr, '\n\tusage: python %s [txt-files] [out-dir] [model-path]\n'%sys.argv[0]
        exit(1)


    # load trained model
    with open(model_path, 'rb') as f:
        dvec = pickle.load(f)
        tag2ind = pickle.load(f)
        crf_model = pickle.load(f)

        ind2tag = { ind:tag for tag,ind in tag2ind.items() }


    files = glob.glob(txt_files)
    N = float(len(files))
    for file_num,txt_file in enumerate(files):

        print 
        print 'predicting: ', txt_file
        print '%d/%d (%.2f%%)' % (file_num+1,N,(1+file_num)/N)
        print 

        ######################################################################
        #                              READ DATA                             #
        ######################################################################

        # read text
        sents = read_txt(txt_file)
        basename = '.'.join(os.path.split(txt_file)[1].split('.')[:-1])

        ######################################################################
        #                            FEATURE ENGINEERING                     #
        ######################################################################

        text_features = extract_features(sents)

        ######################################################################
        #                         FORMATTING DATA                            #
        ######################################################################

        # convert text features to design matrix
        flat_pred_X = dvec.transform(flatten(text_features))

        # reconstruct list structures
        offsets = save_list_structure(text_features)
        pred_X = reconstruct_list( list(flat_pred_X) , offsets)

        ######################################################################
        #                            PREDICTING                              #
        ######################################################################

        # make the predictions
        pred_Y = crf.predict(crf_model, pred_X)
        pred_tags = [ [ind2tag[p] for p in P] for P in pred_Y ]

        assert len(pred_Y) == len(pred_X)
        for i in range(len(pred_Y)):
            assert len(pred_Y[i]) == len(pred_X[i])


        # correct illegal predictions (AKA bad I -> legal B)
        for lineno,preds in enumerate(pred_tags):
            for i in range(len(preds)):
                if preds[i][0] == 'I':
                    if preds[i-1][0] == 'O' or preds[i][1:]!=preds[i][1:]:
                        preds[i] = 'B-%s' % preds[i][2:]

        # assert proper formatting
        for lineno,preds in enumerate(pred_tags):
            for i in range(len(preds)):
                if preds[i][0] == 'I':
                    assert preds[i-1][0] != 'O' and preds[i][1:]==preds[i][1:]

        tags_file = os.path.join(out_dir, '%s.pred' % basename)

        print tags_file

        # output to tags format
        print_predictions(tags_file, sents, pred_tags)




if __name__ == '__main__':
    main()


