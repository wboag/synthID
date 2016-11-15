
import sys
import nltk


def main():

    try:
        txt_file = sys.argv[1]
        tags_file = sys.argv[2]
    except Exception, e:
        print >>sys.stderr, '\n\tusage: python %s [text-file] [tags-file]\n'%sys.argv[0]
        exit(1)


    # read text
    with open(txt_file, 'r') as f:
        lines = [ line.strip() for line in f.readlines() ]

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
            annotations.append( (lineno,start,end,segment,label) )

    # put the real labels back
    seen = set([])
    for lineno,start,end,segment,label in annotations:
        # TODO - ASSUMPTION - assuming each segment is unique on that line
        '''
        key = (lineno,segment)
        assert key not in seen
        seen.add(key)
        '''
        lines[lineno] = lines[lineno].replace(segment, '[**%s**]' % label)

    print '\n'.join(lines)




if __name__ == '__main__':
    main()


