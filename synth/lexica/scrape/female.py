

import re
import os


def main():

    regex = '<tr><td>(\w+)</td><td>(\d+\.\d+)</td><td> [\d,]+ </td><td>(\d+)</td></tr>'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_file = os.path.join(parent_dir, 'web', 'female_names.htm')

    with open(txt_file, 'r') as f:
        text = f.read()

    matches = re.findall(regex, text)

    ranks = [ int(m[2])-1 for m in matches ]
    assert ranks == range(1000)

    # ensure you got it all and output percentages
    for name,percentage,rank in matches:
        print capitalize(name), (float(percentage)/100)



def capitalize(w):
    return w[0].upper() + w[1:].lower()




if __name__ == '__main__':
    main()

