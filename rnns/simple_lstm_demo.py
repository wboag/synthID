import tensorflow as tf
import data_wrapper

batch_size = 20
hidden_units = 32
learning_rate = .005
training_steps = 10**5
empty_tag = 0
outside_tag = 13

graph = tf.Graph()
reader = data_wrapper.DataReader()
output_units = len(reader.tag_index)


def precision(correct_tags,predicted_tags,binary=False):
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

    for sent in range(len(correct_tags)):
        correct_tag_list = reader.decode_tags(correct_tags[sent]).split(' ')
        predicted_tag_list = reader.decode_tags(predicted_tags[sent]).split(' ')
        for tag in range(len(correct_tag_list)):
            predicted = predicted_tag_list[tag]
            actual = correct_tag_list[tag]
            if actual == outside_tag and predicted !=outside_tag:
                false_positive+=1.0
            if binary:
                if actual != outside_tag and actual != empty_tag and predicted ==actual:
                    true_positive+=1.0
            else: 
                if actual != outside_tag and actual != empty_tag and predicted!= empty_tag and predicted != outside_tag:
                    true_positive+=1.0
    return true_positive/(true_positive+false_positive+1e-9)

def recall(correct_tags,predicted_tags,binary=False):
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
    return true_positive/(true_positive+false_negative+1e-9)
def f1(p,r):
    return (2.0*p*r)/(p+r)
with graph.as_default():

    # read batch and look up embeddings
    tokens, tags, lengths = reader.get_train_batch(batch_size)
    inputs = tf.nn.embedding_lookup(tf.Variable(reader.embeddings), tokens)

    # simple LSTM with softmax output
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
    a, _ = tf.nn.dynamic_rnn(cell, inputs, lengths, dtype=tf.float32)
    W = tf.Variable(tf.random_normal([hidden_units, output_units]))
    b = tf.Variable(tf.zeros([output_units]))
    z = tf.matmul(tf.reshape(a, [-1, hidden_units]), W) + b
    z = tf.reshape(z, [-1, tf.shape(a)[1], output_units])

    # calculate loss and backpropogate
    preds = tf.argmax(z, 2)
    x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(z, tags)
    loss = tf.reduce_sum(x_ent) / tf.cast(tf.reduce_sum(lengths), tf.float32)
    step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.initialize_all_variables()


with tf.Session(graph=graph) as session:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord)

    session.run(init)
    for step_num in range(training_steps):
        _, batch_loss = session.run([step, loss])
        if step_num % 100 == 0:
            x, y, y_ = session.run([tokens, tags, preds])
            accuracy = 1.0 * ((y == y_) & (y != 0)).sum() / (y != 0).sum()
            precision_data = precision(y,y_)
            recall_data = recall(y,y_)
      
            # print some info about the batch
            print 'Loss:    ', batch_loss
            print 'Precision:  ', precision_data
            if recall_data!=None:
                print 'Recall: ',recall_data
                print 'f1:   ', f1(recall_data,accuracy)
            print 'Sentence:', reader.decode_tokens(x[0][(y != 0)[0]][:15])
            print 'Truth:   ', reader.decode_tags(y[0][(y != 0)[0]][:15])
            print 'Pred:    ', reader.decode_tags(y_[0][(y != 0)[0]][:15])

    coord.request_stop()
    coord.join(threads)
