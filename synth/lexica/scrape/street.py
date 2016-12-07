

import re
import os


def main():

    #regex = '\.html">([\w ]+)</a></td></tr>'
    regex = '\.html">([\w ]+)</a></td></tr>.<tr><td>(\d+)</td><td>'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_file = os.path.join(parent_dir, 'web', 'most-popular_street_names.html')

    with open(txt_file, 'r') as f:
        text = f.read()

    matches = re.findall(regex, text)

    Z = sum([int(n) for name,n in matches])

    for name,freq in matches:
        print name, float(freq)/Z





if __name__ == '__main__':
    main()

