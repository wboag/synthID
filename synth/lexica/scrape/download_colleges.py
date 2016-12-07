

import re
import os
from commands import getstatusoutput


def main():

    regex = '<a href="/(state-search\.php\?state=\w\w)"'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_file = os.path.join(parent_dir, 'web', 'college.php')

    with open(txt_file, 'r') as f:
        text = f.read()

    matches = re.findall(regex, text)
    print len(matches)

    for match in matches:
        wiki_file = os.path.join(parent_dir, 'web', 'college', match)
        cmd1 = 'wget http://univsearch.com/%s' % match
        cmd2 = 'mv %s %s' % (match,wiki_file)
        getstatusoutput(cmd1)
        getstatusoutput(cmd2)



def capitalize(w):
    return w[0].upper() + w[1:].lower()




if __name__ == '__main__':
    main()

