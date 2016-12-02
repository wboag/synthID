import tensorflow as tf
import data_wrapper
import metrics

batch_size = 64
hidden_units = 32
learning_rate = .005
training_steps = 3*10**2

graph = tf.Graph()
session = tf.Session(graph=graph)
reader = data_wrapper.DataReader()
output_units = len(reader.tag_index)


def get_batch(batch_size, train=False):
    load_batch = reader.get_train_batch if train else reader.get_test_batch
    tokens, tags, lengths = load_batch(batch_size)
    embeddings = tf.Variable(reader.embeddings)
    inputs = tf.nn.embedding_lookup(embeddings, tokens)
    return tokens, tags, lengths, inputs


def predict(inputs, lengths):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
    a, _ = tf.nn.dynamic_rnn(cell, inputs, lengths, dtype=tf.float32)
    W = tf.Variable(tf.random_normal([hidden_units, output_units]))
    b = tf.Variable(tf.zeros([output_units]))
    z = tf.matmul(tf.reshape(a, [-1, hidden_units]), W) + b
    z = tf.reshape(z, [-1, tf.shape(a)[1], output_units])
    preds = tf.argmax(z, 2)
    return z, preds


def cross_entropy(logits, lengths, tags):
    x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tags)
    loss = tf.reduce_sum(x_ent) / tf.cast(tf.reduce_sum(lengths), tf.float32)
    return loss


def evaluate_test_set(session, test_tags, test_preds):
    batch_num = 0
    while True:
        try:
            batch_num += 1
            y, y_ = session.run([test_tags, test_preds])
            precision, recall, f1 = metrics.precision_recall_f1(reader, y, y_)
            print 'precision: ', precision
            print 'recall:    ', recall
            print 'f1:        ', f1
            print
        except tf.errors.OutOfRangeError:
            print 'queue is empty'
            break


with graph.as_default():

    tokens, tags, lengths = reader.get_train_batch(batch_size)
    embeddings = tf.Variable(reader.embeddings)
    inputs = tf.nn.embedding_lookup(embeddings, tokens)

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
    a, _ = tf.nn.dynamic_rnn(cell, inputs, lengths, dtype=tf.float32)
    W = tf.Variable(tf.random_normal([hidden_units, output_units]))
    b = tf.Variable(tf.zeros([output_units]))
    z = tf.matmul(tf.reshape(a, [-1, hidden_units]), W) + b
    z = tf.reshape(z, [-1, tf.shape(a)[1], output_units])

    preds = tf.argmax(z, 2)
    x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(z, tags)
    loss = tf.reduce_sum(x_ent) / tf.cast(tf.reduce_sum(lengths), tf.float32)
    step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.get_variable_scope().reuse_variables()

    ## in the process of refactoring so we dont have to duplicate all of this for testing
    test_tokens, test_tags, test_lengths = reader.get_test_batch(10000)
    test_inputs = tf.nn.embedding_lookup(embeddings, test_tokens)

    # simple LSTM with softmax output
    a_test, _ = tf.nn.dynamic_rnn(cell, test_inputs, test_lengths, dtype=tf.float32)
    z_test = tf.matmul(tf.reshape(a_test, [-1, hidden_units]), W) + b
    z_test = tf.reshape(z_test, [-1, tf.shape(a_test)[1], output_units])
    test_preds = tf.argmax(z_test, 2)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())


with session.as_default():

    session.run(init)
    train_coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, train_coord)

    for step_num in range(training_steps):

        session.run(tokens)
        _, batch_loss = session.run([step, loss])
        if step_num % 50 == 0:
            x, y, y_ = session.run([tokens, tags, preds])
            accuracy = 1.0 * ((y == y_) & (y != 0)).sum() / (y != 0).sum()
            precision, recall, f1 = metrics.precision_recall_f1(reader, y, y_)

            # print some info about the batch
            print 'Loss:      ', batch_loss
            print 'Precision: ', precision
            print 'Recall:    ', recall
            print 'f1:        ', f1
            print 'Sentence:  ', reader.decode_tokens(x[0][(y != 0)[0]][:15])
            print 'Truth:     ', reader.decode_tags(y[0][(y != 0)[0]][:15])
            print 'Pred:      ', reader.decode_tags(y_[0][(y != 0)[0]][:15])
            print

    evaluate_test_set(session, test_tags, test_preds)
    train_coord.request_stop()
    train_coord.join(threads)
