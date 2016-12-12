def precision(reader, correct_tags,predicted_tags,binary=False, counts=False):
    '''Takes in a list of predictions, true tags and computes precision by defining:
    False positive when actual = OUTSIDE and predicted a PHI
    False Negative when actual = PHI and predicted = OUTSIDE
    1) When binary is true :
    True positive when actual = PHI_* and predicted = PHI_*
    2) When binary is false:
    True positive when actual = PHI_X and predicted = PHI_X'''
    true_positive = 0.0
    false_positive = 0.0
    empty_tag = '<PAD>'
    outside_tag = 'OUTSIDE'
    for cat in reader.tag_index:
        per_tag_fp[cat]=0.0
        per_tag_tp[cat]=0.0

    for sent in range(len(correct_tags)):
        correct_tag_list = reader.decode_tags(correct_tags[sent]).split(' ')
        predicted_tag_list = reader.decode_tags(predicted_tags[sent]).split(' ')
        for tag in range(len(correct_tag_list)):
            predicted = predicted_tag_list[tag]
            actual = correct_tag_list[tag]
            if actual == outside_tag and predicted !=outside_tag:
                false_positive+=1.0
                per_tag_fp[predicted]+=1.0
            if binary:
                if actual != outside_tag and actual != empty_tag and predicted ==actual:
                    true_positive+=1.0
            else:
                if actual != outside_tag and actual != empty_tag and predicted!= empty_tag and predicted != outside_tag:
                    true_positive+=1.0
                    per_tag_tp[]
    total = (true_positive+false_positive)
    if counts:
        return true_positive, false_positive
    return true_positive/total if total != 0 else float('nan')


def recall(reader, correct_tags,predicted_tags,binary=False, counts=False):
    '''Takes in a list of predictions and computes recall by defining:
    False positive when actual = OUTSIDE and predicted a PHI_*
    False Negative when actual = PHI_* and predicted = OUTSIDE
    True positive when actual = PHI_X and predicted = PHI_X'''
    true_positive = 0.0
    false_negative = 0.0
    false_positive = 0.0
    empty_tag = '<PAD>'
    outside_tag = 'OUTSIDE'
    #reader = data_wrapper.DataReader()

    for sent in range(len(correct_tags)):
        correct_tag_list = reader.decode_tags(correct_tags[sent]).split(' ')
        predicted_tag_list = reader.decode_tags(predicted_tags[sent]).split(' ')
        for tag in range(len(predicted_tag_list)):
            predicted = predicted_tag_list[tag]
            actual = correct_tag_list[tag]
            if actual == outside_tag and predicted !=outside_tag:
                false_positive+=1.0
            elif actual != outside_tag and actual!= empty_tag and (predicted==empty_tag or predicted== outside_tag):
                false_negative+=1.0
            if binary:
                if actual != outside_tag and actual != empty_tag and predicted!= empty_tag and predicted != outside_tag:
                    true_positive+=1.0
            else:
                if actual != outside_tag and actual != empty_tag and predicted ==actual:
                    true_positive+=1.0
    total = (true_positive+false_negative)
    if counts:
        return true_positive, false_negative
    return true_positive/total if total != 0 else float('nan')

def precision_binary(reader, correct_tags,predicted_tags, counts=False):
    '''Takes in a list of predictions, true tags as 0s and 1s and computes precision by defining:
    False positive when actual = 0 and predicted a 1
    True positive when actual = 1 and predicted = 1
    returns : precision = (true_positive)/(true_positive+false_positive) score between 0.0 and 1.0  else nan if some error happened'''
    true_positive = 0.0
    false_positive = 0.0
    for sent in range(len(correct_tags)):
        correct_tag_list = reader.decode_tags(correct_tags[sent]).split(' ')
        predicted_tag_list = reader.decode_tags(predicted_tags[sent]).split(' ')
        for tag in range(len(correct_tag_list)):
            predicted = predicted_tag_list[tag]
            actual = correct_tag_list[tag]
            if not(actual) and predicted:
                false_positive+=1.0
            if actual and predicted:
                true_positive+=1.0
    total = (true_positive+false_positive)
    if counts:
        return true_positive, false_positive
    return true_positive/total if total != 0 else float('nan')
def recall_binary(reader, correct_tags,predicted_tags, counts=False):
    '''Takes in a list of predictions as 0s and 1s and computes recall by defining:
    False positive when actual = 0 and predicted a 1
    False Negative when actual = 1 and predicted = 0
    True positive when actual = 1 and predicted = 1
    Return recall = (true_positive/true_positive+false_negative)'''
    true_positive = 0.0
    false_negative = 0.0
    false_positive = 0.0
    #reader = data_wrapper.DataReader()

    for sent in range(len(correct_tags)):
        correct_tag_list = reader.decode_tags(correct_tags[sent]).split(' ')
        predicted_tag_list = reader.decode_tags(predicted_tags[sent]).split(' ')
        for tag in range(len(predicted_tag_list)):
            predicted = predicted_tag_list[tag]
            actual = correct_tag_list[tag]
            if not(actual) and predicted:
                false_positive+=1.0
            elif actual and not(predicted):
                false_negative+=1.0
            elif actual and predicted:
                true_positive+=1.0
    total = (true_positive+false_negative)
    if counts:
        return true_positive, false_negative
    return true_positive/total if total != 0 else float('nan')


def f1(p,r):
    '''Takes precision and recall as input and computes f1_score'''
    return (2.0*p*r)/(p+r) if (p + r) != 0 else float('nan')


def precision_recall_f1(reader, correct_tags, predicted_tags, binary=False):
    if not(binary):
        p = precision(reader, correct_tags, predicted_tags, binary=False)
        r = recall(reader, correct_tags, predicted_tags, binary=False)
    else:
        p = precision_binary(reader, correct_tags, predicted_tags)
        r = recall_binary(reader, correct_tags, predicted_tags)
    f = f1(p, r)
    return p, r, f
