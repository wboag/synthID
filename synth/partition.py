
import os
import glob
from commands import getstatusoutput
import random



def main():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    all_data = os.path.join(data_dir, 'all' )

    txt_files       = glob.glob(os.path.join(all_data,'txt'     ,'*.txt' ))
    tags_files      = glob.glob(os.path.join(all_data,'tags'    ,'*.tags'))
    templates_files = glob.glob(os.path.join(all_data,'template','*.txt' ))

    txt_dict       = dictionary(txt_files)
    tags_dict      = dictionary(tags_files)
    templates_dict = dictionary(templates_files)

    train_size = .80
    dev_size   = .10

    train_ind = int(len(txt_files)*train_size)
    dev_ind   = int(len(txt_files)*  dev_size) + train_ind

    random.shuffle(txt_files)

    train_txt_files = txt_files[         :train_ind]
    dev_txt_files   = txt_files[train_ind:  dev_ind]
    test_txt_files  = txt_files[  dev_ind:         ]


    for filename in train_txt_files:
        key = os.path.split(filename)[-1].split('.')[0]
        print 'TRAIN: ', key
        getstatusoutput('cp %s/all/txt/%s %s/train/txt/%s' % (data_dir,txt_dict[key],data_dir,txt_dict[key]))
        getstatusoutput('cp %s/all/tags/%s %s/train/tags/%s' % (data_dir,tags_dict[key],data_dir,tags_dict[key]))
        getstatusoutput('cp %s/all/template/%s %s/train/template/%s' % (data_dir,templates_dict[key],data_dir,templates_dict[key]))


    print


    for filename in dev_txt_files:
        key = os.path.split(filename)[-1].split('.')[0]
        print 'DEV:   ', key
        getstatusoutput('cp %s/all/txt/%s %s/dev/txt/%s' % (data_dir,txt_dict[key],data_dir,txt_dict[key]))
        getstatusoutput('cp %s/all/tags/%s %s/dev/tags/%s' % (data_dir,tags_dict[key],data_dir,tags_dict[key]))
        getstatusoutput('cp %s/all/template/%s %s/dev/template/%s' % (data_dir,templates_dict[key],data_dir,templates_dict[key]))


    print


    for filename in test_txt_files:
        key = os.path.split(filename)[-1].split('.')[0]
        print 'TEST:  ', key
        getstatusoutput('cp %s/all/txt/%s %s/test/txt/%s' % (data_dir,txt_dict[key],data_dir,txt_dict[key]))
        getstatusoutput('cp %s/all/tags/%s %s/test/tags/%s' % (data_dir,tags_dict[key],data_dir,tags_dict[key]))
        getstatusoutput('cp %s/all/template/%s %s/test/template/%s' % (data_dir,templates_dict[key],data_dir,templates_dict[key]))


    



def dictionary(lst):
    d = {}
    for filename in lst:
        key = os.path.split(filename)[-1].split('.')[0]
        d[key] = os.path.split(filename)[-1]
    return d


if __name__ == '__main__':
    main()


