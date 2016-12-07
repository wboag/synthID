

import re
import os


def main():

    # <td width="70%"><a href="/profile.php?id=164492" title="For more information about ANNA MARIA COLLEGE Click here">ANNA MARIA COLLEGE</a></td>
    #regex = '<td width="70%"><a href="\w+" title="For more information about ([\w ]+) Click here">([\w ]+)</a></td>'
    regex = '<td width="70%"><a href=".+" title="For more information about [\w ]+ Click here">([\w ]+)</a></td>'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wiki_dir = os.path.join(parent_dir, 'web', 'college')
    for filename in os.listdir(wiki_dir):

        txt_file = os.path.join(wiki_dir, filename)

        with open(txt_file, 'r') as f:
            text = f.read()

        matches = re.findall(regex, text)

        matches = [ t for t in matches if ('list' not in t[1].lower()) ]

        for name in matches:
            print capitalize(name)


def capitalize(phrase):
    words = phrase.split()
    out = []
    for word in words:
        w = word[0].upper() + word[1:].lower()
        out.append(w)
    return ' '.join(out)



if __name__ == '__main__':
    main()

