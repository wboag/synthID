import tensorflow as tf
import data_wrapper
from datetime import datetime
import metrics

batch_size = 32
hidden_units = 64
learning_rate = .003
training_steps = 10**5
embedding_dim = 50  # 50, 100, 200, or 300

graph = tf.Graph()
session = tf.Session(graph=graph)
reader = data_wrapper.DataReader(embedding_dim=embedding_dim, num_threads=3)
output_units = 2
start_time = datetime.now().strftime('%m-%d-%H-%M-%S')
model_name = 'binary_simple_lstm'


def get_batch(batch_size, train=False):
    load_batch = reader.get_train_batch if train else reader.get_test_batch
    tokens, tags, lengths, filenames, line_nums = load_batch(batch_size)
    binary_tags = get_binary_tags(tags)
    embeddings = tf.get_variable(
        name='embedding_matrix',
        initializer=tf.constant_initializer(reader.embeddings),
        shape=reader.embeddings.shape
    )
    inputs = tf.nn.embedding_lookup(embeddings, tokens)
    return tokens, tags, lengths, inputs, filenames, line_nums, binary_tags

def get_binary_tags(tags):
    six_tensor = 6*tf.ones_like(tags)
    zero_tensor = tf.zeros_like(tags)
    ones_tensor = tf.ones_like(tags)

    is_six = tf.equal(tags, six_tensor)
    is_zero = tf.equal(tags, zero_tensor)
    is_one = tf.equal(tags, ones_tensor)

    six_or_zero = tf.logical_or(is_six, is_zero)
    any_non_phi = tf.logical_or(six_or_zero, is_one)
    any_non_phi = (tf.cast(any_non_phi, dtype=tf.int32) - 1) * -1

    return any_non_phi

def predict(inputs, lengths):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
    a, _ = tf.nn.dynamic_rnn(cell, inputs, lengths, dtype=tf.float32)
    W = tf.Variable(tf.random_normal([hidden_units, output_units]))
    b = tf.Variable(tf.zeros([output_units]))
    z = tf.matmul(tf.reshape(a, [-1, hidden_units]), W) + b
    z = tf.reshape(z, [-1, tf.shape(a)[1], output_units])
    preds = tf.argmax(tf.round(tf.nn.sigmoid(z)),2)
    return z, preds


def cross_entropy(logits, lengths, tags):
    x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tags)
    loss = tf.reduce_sum(x_ent) / tf.cast(tf.reduce_sum(lengths), tf.float32)
    return loss

def evaluate_test_set_binary(session, tags, preds, fnames, lines, batch_limit=None):
    batch_num = 0
    num_sequences = 0
    p_tp_total, p_fp_total, r_tp_total, r_fn_total = 0, 0, 0, 0
    p_tp_total_binary, p_fp_total_binary, r_tp_total_binary, r_fn_total_binary = 0, 0, 0, 0

    while True:
        try:
            #Train binary, eval binary
            y, y_ = session.run([test_binary_tags, test_preds])
            p_tp, p_fp = metrics.precision_binary(reader, y, y_, counts=True)
            r_tp, r_fn = metrics.recall_binary(reader, y, y_, counts=True)
            p_tp_total += p_tp
            p_fp_total += p_fp
            r_tp_total += r_tp
            r_fn_total += r_fn

            num_sequences += len(y)
            batch_num += 1
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
        tokens, tags, lengths, inputs, fnames, lines, binary_tags = \
            get_batch(batch_size, train=True)
        
        logits, preds = predict(inputs, lengths)
        loss = cross_entropy(logits, lengths, binary_tags)
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.variable_scope('rnn', reuse=True):
        test_tokens, test_tags, test_lengths, test_inputs, \
            test_fnames, test_lines, test_binary_tags = get_batch(10000, train=False)
        test_logits, test_preds = predict(test_inputs, test_lengths)
        test_loss = cross_entropy(test_logits, test_lengths, test_binary_tags)

    saver = tf.train.Saver()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

with session.as_default():
    session.run(init)
    train_coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, train_coord)

    warm_start_init_step = 0
    if warm_start_init_step != 0:
        ckpt_file = 'checkpoints/{}-{}'.format(model_name, warm_start_init_step)
        saver.restore(session, ckpt_file)

    for step_num in range(training_steps):

        _, batch_loss, filenames, line_nums = \
            session.run([step, loss, fnames, lines])

        # logging to stdout for sanity checks every 50 steps
        if step_num % 50 == 0:
            x, y, y_ = session.run([tokens, binary_tags, preds])
            accuracy = 1.0 * ((y == y_) & (y != 0)).sum() / (y != 0).sum()
            precision = metrics.precision_binary(reader, y, y_)
            recall = metrics.recall_binary(reader, y, y_)
            f1 = metrics.f1(precision, recall)

            # print some info about the batch
            print 'Loss:      ', batch_loss
            print 'Precision: ', precision
            print 'Recall:    ', recall
            print 'f1:        ', f1
            print 'Sentence:  ', reader.decode_tokens(x[0,:15])
            print 'Truth:     ', y[0,:15]
            print 'Pred:      ', y_[0,:15]
            print

        # write train accuracy to log files every 100 steps
        if step_num % 100 == 0:
            train_loss = 0
            train_eval_size = 50
            for i in range(train_eval_size):
                train_loss += session.run([loss])[0]
            train_loss = train_loss / float(train_eval_size)
            with open('logs/train_log-{}.txt'.format(start_time), 'a') as log:
                log.write('{} {}\n'.format(step_num, train_loss))

        # save model parameters every 1000 steps
        if step_num % 1000 == 0 and step_num > 0:
            saver.save(session, 'checkpoints/{}'.format(model_name), global_step=step_num)

    evaluate_test_set_binary(session, test_binary_tags, test_preds, test_fnames, test_lines)
    train_coord.request_stop()
    train_coord.join(threads)
