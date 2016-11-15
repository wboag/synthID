

import re
import os


def main():

    #regex = '<tr><td><a href=http://names.mongabay.com/baby-names/application/rank-M-US-(\w+).html>JAMES</td><td>3.318</td><td> 4,840,833 </td><td>1</td></tr>'
    regex = '<tr><td><a href=http://names.mongabay.com/baby-names/application/rank-M-US-(\w+).html>\w+</td><td>(\d+\.\d+)</td><td> [\d,]+ </td><td>(\d+)</td></tr>'
    #regex = '<tr><td><a href=http://names.mongabay.com/baby-names/application/rank-M-US-(\w+).html>'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_file = os.path.join(parent_dir, 'web', 'male_names.htm')

    with open(txt_file, 'r') as f:
        text = f.read()

    matches = re.findall(regex, text)

    ranks = [ int(m[2])-1 for m in matches ]
    assert ranks == range(300)

    # ensure you got it all and output percentages
    for name,percentage,rank in matches:
        print capitalize(name), (float(percentage)/100)



def capitalize(w):
    return w[0].upper() + w[1:].lower()




if __name__ == '__main__':
    main()

