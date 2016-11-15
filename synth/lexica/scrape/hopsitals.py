

import re
import os


def main():

    regex = '<li><a href="[^"]+" class="new" title="[^"]+">([^<]+)</a>'
    # <li><a href="/w/index.php?title=Franciscan_Hospital_for_Children&amp;action=edit&amp;redlink=1" class="new" title="Franciscan Hospital for Children (page does not exist)">Franciscan Hospital for Children</a></li>

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wiki_dir = os.path.join(parent_dir, 'web', 'wiki')
    for filename in os.listdir(wiki_dir):

        txt_file = os.path.join(wiki_dir, filename)

        with open(txt_file, 'r') as f:
            text = f.read()

        matches = re.findall(regex, text)

        # ensure you got it all and output percentages
        for name in matches:
            if '(' in name:
                continue
            if ' - ' in name:
                continue
            if ' in ' in name:
                continue
            print name.strip('.')




def capitalize(w):
    return w[0].upper() + w[1:].lower()




if __name__ == '__main__':
    main()

