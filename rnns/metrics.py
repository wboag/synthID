
def initialize_dicts(tag_list,fp=True,tp=True,fn=False,overall=True):
    '''Helper method used to initialize per_category precision and recall dictionaries'''
    per_tag_fp = {}
    per_tag_tp = {}
    per_tag_overall = {}
    if fn:
        per_tag_fn = {}
    for t in tag_list:
        per_tag_fp[t]=0.0
        per_tag_tp[t]=0.0
        per_tag_overall[t]=0.0
        if fn:
            per_tag_fn[t]=0.0
    if fn:
        return per_tag_fp,per_tag_tp,per_tag_fn,per_tag_overall
    else:
        return per_tag_fp,per_tag_tp,per_tag_overall

def precision_tag(reader, correct_tags,predicted_tags):
    '''
    Takes in a list of predictions, true tags and computes precision per tag by defining:
    False positive when actual = OUTSIDE and predicted a PHI
    False Negative when actual = PHI and predicted = OUTSIDE
    True positive when actual = PHI_Cat1 and predicted = PHI_Cat1
    Compute precision per category and returns a dict with keys :
    ['DATE','HOSPITAL','LOCATION','CONTACT','NUMBER','NAME','AVERAGE']
    And precision value for each category
    '''
    true_positive = 0.0
    false_positive = 0.0
    empty_tag = '<PAD>'
    outside_tag = 'OUTSIDE'
    tag_list = ['DATE','HOSPITAL','LOCATION','CONTACT','NUMBER','NAME']
    per_tag_fp,per_tag_tp,per_tag_p = initialize_dicts(tag_list)

    for sent in range(len(correct_tags)):
        correct_tag_list = reader.decode_tags(correct_tags[sent]).split(' ')
        predicted_tag_list = reader.decode_tags(predicted_tags[sent]).split(' ')
        for tag in range(len(correct_tag_list)):
            predicted = predicted_tag_list[tag]
            actual = correct_tag_list[tag]
            if actual == outside_tag and predicted !=outside_tag:
                false_positive+=1.0
                for t in tag_list:
                    if t in actual:
                        per_tag_fp[t]+=1.0
            if actual != outside_tag and actual != empty_tag and are_same_categories(actual,predicted):
                true_positive+=1.0
                for t in tag_list:
                    if t in actual:
                        per_tag_tp[t]+=1.0
    total = (true_positive+false_positive)
    for t in tag_list:
        total_tag = per_tag_tp[t]+per_tag_fp[t]
        per_tag_p[t]=per_tag_tp[t]/total_tag if total_tag!=0 else float('nan')
    per_tag_p['AVERAGE']= true_positive/total if total !=0 else float('nan')
    return per_tag_p

def recall_tag(reader, correct_tags,predicted_tags):
    '''Takes in a list of predictions and computes recall by defining:
    False positive when actual = OUTSIDE and predicted a PHI_*
    False Negative when actual = PHI_* and predicted = OUTSIDE
    True positive when actual = PHI_Cat1 and predicted = PHI_Cat1 
    Returns a dictionary of recall values among the following:
    ['DATE','HOSPITAL','LOCATION','CONTACT','NUMBER','NAME','AVERAGE']
    '''
    true_positive = 0.0
    false_negative = 0.0
    false_positive = 0.0
    empty_tag = '<PAD>'
    outside_tag = 'OUTSIDE'
    tag_list = ['DATE','HOSPITAL','LOCATION','CONTACT','NUMBER','NAME']
    per_tag_fp,per_tag_tp,per_tag_fn,per_tag_r = initialize_dicts(tag_list,True,True,True,True)
  
    
    #reader = data_wrapper.DataReader()

    for sent in range(len(correct_tags)):
        correct_tag_list = reader.decode_tags(correct_tags[sent]).split(' ')
        predicted_tag_list = reader.decode_tags(predicted_tags[sent]).split(' ')
        for tag in range(len(predicted_tag_list)):
            predicted = predicted_tag_list[tag]
            actual = correct_tag_list[tag]
            if actual == outside_tag and predicted !=outside_tag:
                false_positive+=1.0
                for t in tag_list:
                    if t in actual:
                        per_tag_fp[t]+=1.0
            elif actual != outside_tag and actual!= empty_tag and (predicted==empty_tag or predicted== outside_tag):
                false_negative+=1.0
                for t in tag_list:
                    if t in actual:
                        per_tag_fn[t]+=1.0
            if actual != outside_tag and actual != empty_tag and are_same_categories(predicted,actual):
                true_positive+=1.0
                for t in tag_list:
                    if t in actual:
                        per_tag_fp[t]+=1.0
    total = (true_positive+false_negative)
    for t in tag_list:
        total_tag = per_tag_tp[t]+per_tag_fn[t]
        per_tag_r[t]=per_tag_tp[t]/total_tag if total_tag!=0 else float('nan')
    per_tag_r['AVERAGE']= true_positive/total if total !=0 else float('nan')
    return per_tag_r




def precision(reader, correct_tags,predicted_tags,binary=False, counts=False):
    '''Takes in a list of predictions, true tags and computes precision by defining:
    False positive when actual = OUTSIDE and predicted a PHI
    False Negative when actual = PHI and predicted = OUTSIDE
    1) When binary is true :
    True positive when actual = PHI_* and predicted = PHI_*
    2) When binary is false:
    True positive when actual = PHI_X and predicted = PHI_X
    Returns : Precision score = (tp)/(tp+fp) or 'nan'
    '''
    true_positive = 0.0
    false_positive = 0.0
    empty_tag = '<PAD>'
    outside_tag = 'OUTSIDE'
    for sent in range(len(correct_tags)):
        correct_tag_list = reader.decode_tags(correct_tags[sent]).split(' ')
        predicted_tag_list = reader.decode_tags(predicted_tags[sent]).split(' ')
        for tag in range(len(correct_tag_list)):
            predicted = predicted_tag_list[tag]
            actual = correct_tag_list[tag]
            if actual == outside_tag and predicted !=outside_tag:
                false_positive+=1.0
            if binary:
                if actual != outside_tag and actual != empty_tag and predicted!= empty_tag and predicted != outside_tag:
                    true_positive+=1.0
            else:
                if actual != outside_tag and actual != empty_tag and predicted ==actual:
                    true_positive+=1.0
    total = (true_positive+false_positive)
    if counts:
        return true_positive, false_positive
    return true_positive/total if total != 0 else float('nan')


def recall(reader, correct_tags,predicted_tags,binary=False, counts=False):
    '''Takes in a list of predictions and computes recall by defining:
    False positive when actual = OUTSIDE and predicted a PHI_*
    False Negative when actual = PHI_* and predicted = OUTSIDE
    True positive when actual = PHI_X and predicted = PHI_X
    Returns : Recall score = (tp/(tp+fn)) or 'nan'
    '''
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
        correct_tag_list = correct_tags[sent]
        predicted_tag_list = predicted_tags[sent]
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
        correct_tag_list = correct_tags[sent]
        predicted_tag_list = predicted_tags[sent]

        for tag in range(len(correct_tag_list)):
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

def are_same_categories(tag1,tag2):
    '''Takes 2 tags and return true if they are of the same category where categories are :
    -DATE, HOSPITAL, LOCATION, CONTACT, NUMBER, NAME'''
    test_1 = tag1==tag2 
    test_2 = tag1[:-2]==tag2[:-2]
    test_3 = False
    tag_list = ['DATE','HOSPITAL','LOCATION','CONTACT','NUMBER','NAME']
    for t in tag_list:
        if t in tag1:
            test_3= t in tag2
    return test_1 or test_2 or test_3

def f1(p,r):
    '''Takes precision and recall as input and computes f1_score'''
    return (2.0*p*r)/(p+r) if (p + r) != 0 else float('nan')

def f1_tags(p_dict,r_dict):
    '''Takes precision_dict and recall_dict as input and computes f1_score for each tag along with an average f1 score'''
    f1_dict = {}
    for k in p_dict.keys():
        f1_dict[k]=f1(p_dict[k],r_dict[k])
    return f1_dict

def precision_recall_f1(reader, correct_tags, predicted_tags, binary=False):
    '''Given correct_tags and predicted_tags , computes precision,f1 and recall and return them'''
    if not(binary):
        p = precision(reader, correct_tags, predicted_tags, binary=False)
        r = recall(reader, correct_tags, predicted_tags, binary=False)
    else:
        p = precision_binary(reader, correct_tags, predicted_tags)
        r = recall_binary(reader, correct_tags, predicted_tags)
    f = f1(p, r)
    return p, r, f

def precision_recall_f1_tags(reader, correct_tags, predicted_tags):
    '''Given correct_tags and predicted_tags , computes precision,f1 and recall per tag and return precision,recall and f1 dictionaries'''
    p_dict = precision_tag(reader, correct_tags, predicted_tags)
    r_dict = recall_tag(reader, correct_tags, predicted_tags)
    f1_dict = f1_tags(p_dict, r_dict)
    return p_dict, r_dict, f1_dict
