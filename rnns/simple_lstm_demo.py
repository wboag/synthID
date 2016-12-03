import tensorflow as tf
import data_wrapper
import metrics

batch_size = 64
hidden_units = 32
learning_rate = .005
training_steps = 10**3
embedding_dim = 100  # 50, 100, 200, or 300

graph = tf.Graph()
session = tf.Session(graph=graph)
reader = data_wrapper.DataReader(embedding_dimension=embedding_dim)
output_units = len(reader.tag_index)


def get_batch(batch_size, train=False):
    load_batch = reader.get_train_batch if train else reader.get_test_batch
    tokens, tags, lengths = load_batch(batch_size)
    embeddings = tf.get_variable(
        name='embedding_matrix',
        initializer=tf.constant_initializer(reader.embeddings),
        shape=reader.embeddings.shape
    )
    inputs = tf.nn.embedding_lookup(embeddings, tokens)
    return tokens, tags, lengths, inputs


def predict(inputs, lengths):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
    a, _ = tf.nn.dynamic_rnn(cell, inputs, lengths, dtype=tf.float32)
    W = tf.get_variable(
        name='fc_weights',
        initializer=tf.random_normal_initializer(),
        shape=[hidden_units, output_units]
    )
    b = tf.get_variable(
        name='fc_biases',
        initializer=tf.constant_initializer(),
        shape=[output_units]
    )
    z = tf.matmul(tf.reshape(a, [-1, hidden_units]), W) + b
    z = tf.reshape(z, [-1, tf.shape(a)[1], output_units])
    preds = tf.argmax(z, 2)
    return z, preds


def cross_entropy(logits, lengths, tags):
    x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tags)
    loss = tf.reduce_sum(x_ent) / tf.cast(tf.reduce_sum(lengths), tf.float32)
    return loss


def evaluate_test_set(session, tags, preds, batch_limit=None):
    batch_num = 0
    num_sequences = 0
    p_tp_total, p_fp_total, r_tp_total, r_fn_total = 0, 0, 0, 0
    while True:
        try:
            y, y_ = session.run([tags, preds])
            p_tp, p_fp = metrics.precision(reader, y, y_, counts=True)
            r_tp, r_fn = metrics.recall(reader, y, y_, counts=True)
            p_tp_total += p_tp
            p_fp_total += p_fp
            r_tp_total += r_tp
            r_fn_total += r_fn
            batch_num += 1
            num_sequences += len(y)
            if batch_num == batch_limit:
                break
        except tf.errors.OutOfRangeError:
            print 'test queue is empty'
            break
    if p_tp_total:
        precision = p_tp_total / (p_tp_total + p_fp_total)
        recall = r_tp_total / (r_tp_total + r_fn_total)
        f1 = metrics.f1(precision, recall)

        print 'Evaluated {} sequences from test set'.format(num_sequences)
        print 'Precision:  ', precision
        print 'Recall:     ', recall
        print 'f1:         ', f1


with graph.as_default():

    with tf.variable_scope('rnn'):
        tokens, tags, lengths, inputs = get_batch(batch_size, train=True)
        logits, preds = predict(inputs, lengths)
        loss = cross_entropy(logits, lengths, tags)
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.variable_scope('rnn', reuse=True):
        test_tokens, test_tags, test_lengths, test_inputs = get_batch(10000)
        test_logits, test_preds = predict(test_inputs, test_lengths)
        test_loss = cross_entropy(test_logits, test_lengths, test_tags)

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

    evaluate_test_set(session, test_tags, test_preds, batch_limit=3)
    train_coord.request_stop()
    train_coord.join(threads)
