

import os



def read_names(filename):
    names  = []
    scores = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            toks = line.strip().split()
            names.append(toks[0])
            scores.append(float(toks[1]))

    Z = sum(scores)
    pdf = [ s/Z for s in scores ]

    previous = 0
    cdf = []
    for p in pdf:
        previous += p
        cdf.append(previous)

    # ENSURE numeric issues dont fuck up the sums-to-one property
    cdf[-1] = 1.0

    return list(zip(names,cdf))




def read_unweighted(filename):
    hospitals = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            hospitals.append(line.strip())

    N = len(hospitals)
    cdf = [ float(i)/N for i in range(N) ]

    # ENSURE numeric issues dont fuck up the sums-to-one property
    cdf[-1] = 1.0

    return list(zip(hospitals,cdf))




###########################################################################
#                              Generate Info                              #
###########################################################################


lexica_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'lexica','final')


# read names
male_names   = read_names(os.path.join(lexica_dir, 'names-male.txt'   ))
female_names = read_names(os.path.join(lexica_dir, 'names-female.txt' ))
last_names   = read_names(os.path.join(lexica_dir, 'names-surname.txt'))


# misc
hospitals = read_unweighted(os.path.join(lexica_dir, 'hospitals.txt'))
companies = read_unweighted(os.path.join(lexica_dir, 'companies.txt'))
locations = read_unweighted(os.path.join(lexica_dir, 'locations.txt'))

