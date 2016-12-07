

import re
import os


def main():

    regex = '<tr>\n<td><a href="/wiki/\w+" title="[\w ]+">[\w ]+</a></td>\n<td>(\w+\.\w+)</td>\n<td>\d+</td>\n<td>\d+</td>'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_file = os.path.join(parent_dir, 'web', 'List_of_most_popular_websites')

    with open(txt_file, 'r') as f:
        text = f.read()

    matches = re.findall(regex, text)

    # ensure you got it all and output percentages
    for url in matches:
        print url



def capitalize(w):
    return w[0].upper() + w[1:].lower()




if __name__ == '__main__':
    main()

