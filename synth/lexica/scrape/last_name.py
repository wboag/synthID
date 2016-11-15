

import re
import os


def main():

    regex = '<tr><td>(\w+)<td class="c\d+">[\d,]+<td class="c\d+">\d+<td class="c\d+">\d+\.\d+%<td class="c\d+">(\d+)<td class="c\d+">([\d,]+)</td></tr>'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_file = os.path.join(parent_dir, 'web', 'last_names.htm')

    with open(txt_file, 'r') as f:
        text = f.read()

    matches = re.findall(regex, text)

    ranks = [ int(m[1])-1 for m in matches ]

    Z = sum([ get_num(freq) for name,rank,freq in matches ])

    # ensure you got it all and output percentages
    for name,rank,freq in matches:
        print capitalize(name), get_num(freq)/Z



def get_num(string):
    num_string = string.replace(',', '')
    return float(num_string)



def capitalize(w):
    return w[0].upper() + w[1:].lower()




if __name__ == '__main__':
    main()

