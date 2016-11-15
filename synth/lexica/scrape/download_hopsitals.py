

import re
import os
from commands import getstatusoutput


def main():

    regex = '<li><a href="/wiki/(List_of_hospitals_in_[\w_]+)" title="List of hospitals in [\w\s]+">[\w\s]+</a></li>'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_file = os.path.join(parent_dir, 'web', 'Lists_of_hospitals_in_the_United_States')

    with open(txt_file, 'r') as f:
        text = f.read()

    matches = re.findall(regex, text)

    for match in matches:
        wiki_file = os.path.join(parent_dir, 'web', 'wiki', match)
        cmd1 = 'wget https://en.wikipedia.org/wiki/%s' % match
        cmd2 = 'mv %s %s' % (match,wiki_file)
        getstatusoutput(cmd1)
        getstatusoutput(cmd2)



def capitalize(w):
    return w[0].upper() + w[1:].lower()




if __name__ == '__main__':
    main()

