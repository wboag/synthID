

import psycopg2
import pandas as pd
import re
import random
from collections import defaultdict
import nltk
import sys
import os


import phi


random.seed(5)


def main():

    # connect to mimic database
    con = psycopg2.connect(dbname='mimic')

    #base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = '/scratch/wboag/synthid'
    data_dir = os.path.join(base_dir, 'data', 'final', 'all')

    # query for text
    #query = 'select subject_id,text from mimiciii.noteevents where subject_id=88472;'
    #query = 'select subject_id,text from mimiciii.noteevents where subject_id=1056;'
    query = 'select subject_id,text from mimiciii.noteevents;'
    val = pd.read_sql_query(query, con)

    iteration = 0

    # search over all text
    subj_counts = defaultdict(int)
    for subject_id,text in zip(val['subject_id'],val['text']):
        '''
        # show data
        print
        print text
        print 
        '''

        iteration += 1
        if iteration >= 8000: break

        # unique identifier for this note
        note_number = subj_counts[subject_id]
        subj_counts[subject_id] += 1


        name = 'record-%d-%d' % (subject_id,note_number)
        if os.path.exists('%s/txt/%s.txt' % (data_dir,name)):
            continue

        print subject_id, note_number

        #tags = re.findall('(\[\*\*[^\]]*\*\*\])', text)
        #print tags
        #print 

        '''
        print '\n'
        print text
        print '\n'
        '''

        # re-id'd text
        reid_lines = []

        # idenitifying information
        annotations = []

        info = []

        # go through line-by-line (easier annotations)
        for lineno,line in enumerate(text.split('\n')):

            #if lineno != 25: continue

            '''
            tags = re.findall('(\[\*\*[^\]]*\*\*\])', line)
            if len(tags) < 2: continue
            '''

            '''
            print '----------------'
            print line
            '''

            S = line

            # replace tag with a single token
            tags = re.findall('(\[\*\*[^*]*\*\*\])', S)
            for tag in tags:
                normed_tag = tag.replace(' ','_')
                S.replace(tag, normed_tag)

            # tokenization (should really use tokenizer
            toks = S.split()

            # repeat this process until all tags have been substituted on this line
            for ind,token in enumerate(toks):

                if not token.startswith('[**'):
                    continue

                '''
                print 
                print 'S: ', S
                '''

                # get starting index for first tag so you can get
                #  ...text... [** tag **] ...rest...
                start = S.index('[**')
                end   = start + S[start:].index('**]') + 3

                # find the relevant spans
                prefix = S[     :start]
                tag    = S[start:end  ]
                rest   = S[end:       ]

                label = tag[3:-3].strip()

                if label == '':
                    S = '%s%s' % (prefix,rest)
                    continue

                '''
                print 
                print tag
                print S[max(0,start-25):end+25]
                print 
                exit()
                '''

                # entity replacement
                #print 'iter: ', iteration
                new_entity,simple_label = generate_new_entity(label, prefix, rest)
                if simple_label != 'DATE':
                    info.append( (line,label,simple_label,new_entity) )
                    '''
                    print line
                    print 'old label: ', label
                    print 'new label: ', simple_label
                    print new_entity
                    print
                    '''
                #exit()

                #sys.stdin.readline()
                #exit()
                #suffix = re.sub('[^a-zA-Z0-9]','',label).lower()
                #new_entity = 'John%s Smith%s' % (suffix,suffix)
                new_size = len(tokenizer_corrections(nltk.word_tokenize(new_entity)))

                # count how many tokens came before the tag (so you know its position)
                tokens = tokenizer_corrections(nltk.word_tokenize(prefix.strip() + ' ' + new_entity))

                '''
                print 'prefix: [%s]' % prefix
                print 'tag:    [%s]' % tag
                print 'rest:   [%s]' % rest

                print 'tokens: ' , tokens
                '''

                # where this token span begins
                position = len(tokens)

                # make new sentence
                S = '%s %s %s' % (prefix,new_entity,rest)
                #S = '%s%s%s' % (prefix,new_entity,rest)

                '''
                print 'new: ', S
                print
                '''

                # add this to the list of annotations
                #annotation = (lineno, position, position+new_size-1, new_entity, label)
                annotation = (lineno, position-new_size, position-1, new_entity, simple_label)
                annotations.append(annotation)

                #exit()

            #print S
            reid_lines.append(S)

        '''
        for line,label,simple_label,new_entity in sorted(info,key=lambda t:t[2]):
            print line
            print 'old label: ', label
            print 'new label: ', simple_label
            print new_entity
            print
        print '\n\n'
        continue
        exit()
        '''

        #continue

        labels = set([ ann[-1] for ann in annotations ])
        #print labels
        #exit()

        # assemble re-id'd text
        reid_text = '\n'.join(reid_lines)
        #print reid_text

        # ensure directories exist
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            os.mkdir(os.path.join(data_dir, 'template'))
            os.mkdir(os.path.join(data_dir, 'txt'     ))
            os.mkdir(os.path.join(data_dir, 'tags'    ))

        # output template text to file
        with open('%s/template/%s.txt' % (data_dir,name), 'w') as f:
            print >>f, text

        # output re-id text to file
        with open('%s/txt/%s.txt' % (data_dir,name), 'w') as f:
            print >>f, reid_text

        # out annotations to file
        with  open('%s/tags/%s.tags' % (data_dir,name), 'w') as f:
            for lineno,start,end,entity,label in annotations:
                '''
                print 
                print subject_id, note_number
                print lineno
                print reid_lines[lineno]
                print nltk.word_tokenize(reid_lines[lineno])
                print start
                print end
                print entity
                print tokenizer_corrections(nltk.word_tokenize(entity))
                print nltk.word_tokenize(reid_lines[lineno])[start:end+1]
                print label
                print 
                '''
                assert tokenizer_corrections(nltk.word_tokenize(entity)) ==  nltk.word_tokenize(reid_lines[lineno])[start:end+1]
                print >>f, '%d:%d\t%d:%d\ttext=[[%s]]\tlabel=[[%s]]' % (lineno,start,lineno,end,entity,label)




def generate_new_entity(old_label, prefix, rest):

    #print 'label: ', old_label
    low_label = old_label.lower()

    date = is_date(low_label)

    #print '\t', low_label

    # dates
    if date:
        new_entity = date
        label = 'DATE'
    elif 'holiday' in low_label:
        new_entity = generate_holiday(low_label)
        label = 'DATE'
    elif 'hospital' in low_label:
        new_entity = generate_hospital(low_label)
        label = 'HOSPITAL'
    elif 'location' in low_label:
        new_entity = generate_location(low_label)
        label = 'LOCATION'
    elif 'name' in low_label:
        new_entity = generate_name(low_label)
        label = 'NAME'
    elif 'dictator info' in low_label:
        new_entity = generate_name(low_label)
        label = 'NAME'
    elif 'contact info' in low_label:
        new_entity = generate_name(low_label)
        label = 'NAME'
    elif 'attending info' in low_label:
        new_entity = generate_name(low_label)
        label = 'NAME'
    elif 'telephone' in low_label:
        nums = [ random.randint(0,9) for _ in range(10) ]
        new_entity = '%d%d%d-%d%d%d-%d%d%d%d' % tuple(nums)
        label = 'CONTACT'

    elif 'job number' in low_label:
        new_entity = old_label.split()[-1]
        label = 'NUMBER'
    elif 'md number' in low_label:
        new_entity = old_label.split()[-1]
        label = 'NUMBER'
    elif 'pager number' in low_label:
        new_entity = old_label.split()[-1]
        label = 'NUMBER'
    elif 'medical record number' in low_label:
        new_entity = old_label.split()[-1]
        label = 'NUMBER'
    elif 'numeric identifier' in low_label:
        new_entity = old_label.split()[-1]
        label = 'NUMBER'
    elif 'provider number' in low_label:
        new_entity = old_label.split()[-1]
        label = 'NUMBER'
    elif 'clip number' in low_label:
        new_entity = old_label.split()[-1]
        label = 'NUMBER'
    elif 'serial number' in low_label:
        new_entity = old_label.split()[-1]
        label = 'NUMBER'
    elif 'social security number' in low_label:
        new_entity = old_label.split()[-1]
        label = 'NUMBER'
    elif 'unit number' in low_label:
        new_entity = '12'
        label = 'NUMBER'

    elif 'url ' in low_label:
        new_entity = generate_url(low_label)
        label = 'CONTACT'
    elif 'university' in low_label:
        new_entity = generate_college(low_label)
        label = 'LOCATION'
    elif ('address' in low_label) or ('po box' in low_label):
        new_entity = generate_address(low_label)
        label = 'LOCATION'
    elif 'state' in low_label:
        new_entity = generate_state(low_label)
        label = 'LOCATION'
    elif 'country' in low_label:
        new_entity = generate_country(low_label)
        label = 'LOCATION'
    elif 'company' in low_label:
        new_entity = generate_company(low_label)
        label = 'LOCATION'

    elif 'age over 90' in low_label:
        new_entity = low_label
        label = 'NUMBER'
    elif re.search('^\d+$', low_label):
        new_entity = low_label
        label = 'NUMBER'
    elif re.search('^[\d-]+$', low_label):
        new_entity = low_label
        label = 'NUMBER'
    elif re.search('^[-\d/]+$', low_label):
        new_entity = low_label
        label = 'NUMBER'
    else:
        print 'UNKNOWN: ', label
        exit(1)

    return new_entity, label



def is_date(string):
    string = string.lower()
    if re.search('^\d\d\d\d-\d\d?-\d\d?$', string): return string
    if re.search('^\d\d?-\d\d?$'         , string): return string
    if re.search('^\d\d\d\d$'            , string): return string
    if re.search('^\d\d?/\d\d\d\d$'      , string): return string
    if re.search('^\d-/\d\d\d\d$'        , string): return string[0]+string[2:]
    if re.search('january'               , string): return string
    if re.search('february'              , string): return string
    if re.search('march'                 , string): return string
    if re.search('april'                 , string): return string
    if re.search('may'                   , string): return string
    if re.search('june'                  , string): return string
    if re.search('july'                  , string): return string
    if re.search('august'                , string): return string
    if re.search('september'             , string): return string
    if re.search('october'               , string): return string
    if re.search('november'              , string): return string
    if re.search('december'              , string): return string
    if re.search('month'                 , string): return 'July'
    if re.search('year'                  , string): return '2012'
    if re.search('date range'            , string): return 'July - September'
    return False




def generate_name(label):
    new_name = []

    # [OPTIONAL] if first name is there, add it
    if 'first' in label:
        r = random.random()
        name = find(phi.male_names, r)
        new_name.append(name[0])

    # if no first name OR last name is specified, add it
    if len(new_name)==0 or 'last' in label:
        r = random.random()
        name = find(phi.last_names, r)
        new_name.append(name[0])

    return ' '.join(new_name)



def generate_url(label):
    r = random.random()
    name = find(phi.urls, r)
    return name[0]



def generate_holiday(label):
    r = random.random()
    name = find(phi.holidays, r)
    return name[0]



def generate_location(label):
    r = random.random()
    name = find(phi.locations, r)
    return name[0]



def generate_college(label):
    r = random.random()
    name = find(phi.colleges, r)
    return name[0]



def generate_hospital(label):
    r = random.random()
    name = find(phi.hospitals, r)
    return name[0]



def generate_address(label):
    r = random.random()
    addr = find(phi.streets, r)[0] # street name
    if ('apartment address' in label) or ('street address' in label):
        num = random.randint(1,160)
        addr = '%d %s' % (num,addr)
    return addr



def generate_company(label):
    r = random.random()
    name = find(phi.companies, r)
    return name[0]



def generate_country(label):
    r = random.random()
    name = find(phi.countries, r)
    return name[0]



def generate_state(label):
    r = random.random()
    name = find(phi.states, r)
    return name[0]



def find(pairs, r):
    N = len(pairs)
    if N == 1:
        return pairs[0]

    pivot = pairs[N/2][1]
    if r <= pivot:
        return find(pairs[:N/2], r)
    else:
        return find(pairs[N/2:], r)



def tokenizer_corrections(toks):
    if toks[-1] == '.':
        del toks[-1]
        toks[-1] = toks[-1] + '.'
    return toks



if __name__ == '__main__':
    main()


