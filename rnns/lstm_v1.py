import tensorflow as tf
import data_wrapper
import numpy as np


batch_size = 128
test_batch_size = 10000
hidden_units = 32
learning_rate = .005
training_steps = 10**5

graph = tf.Graph()
reader = data_wrapper.DataReader()
output_units = len(reader.tag_index)

with graph.as_default():


    ## TRAINING
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

    tf.get_variable_scope().reuse_variables()

    ## TESTING
    test_tokens, test_tags, test_lengths = reader.get_test_batch(test_batch_size)
    test_inputs = tf.nn.embedding_lookup(tf.Variable(reader.embeddings), test_tokens)

    # simple LSTM with softmax output
    a_test, _ = tf.nn.dynamic_rnn(cell, test_inputs, test_lengths, dtype=tf.float32)
    z_test = tf.matmul(tf.reshape(a_test, [-1, hidden_units]), W) + b
    z_test = tf.reshape(z_test, [-1, tf.shape(a_test)[1], output_units])

    # calculate loss and backpropogate
    test_preds = tf.argmax(z_test, 2)
    test_x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(z_test, test_tags)
    test_loss = tf.reduce_sum(test_x_ent) / tf.cast(tf.reduce_sum(test_lengths), tf.float32)

    init = tf.initialize_all_variables()
    init2 = tf.initialize_local_variables()

with tf.Session(graph=graph) as session:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord)

    session.run([init, init2])

    for step_num in range(training_steps):
        session.run(init2)
        _, batch_loss = session.run([step, loss])
        if step_num % 100 == 0:
            x, y, y_ = session.run([tokens, tags, preds])
            accuracy = 1.0 * ((y == y_) & (y != 0)).sum() / (y != 0).sum()

            # print some info about the batch
            print 'Loss:    ', batch_loss
            print 'Accuracy:', accuracy
            print 'Sentence:', reader.decode_tokens(x[0][(y != 0)[0]][:15])
            print 'Truth:   ', reader.decode_tags(y[0][(y != 0)[0]][:15])
            print 'Pred:    ', reader.decode_tags(y_[0][(y != 0)[0]][:15])
            print

    print('Testing')

    test_real, test_pred, test_loss = session.run([test_tags, test_preds, test_loss])
    test_acc = 1.0 * ((test_real == test_pred) & (test_real != 0)).sum() / (test_real != 0).sum()

    print 'Test Loss: ', test_loss
    print 'Test Accuracy: ', test_acc
    print

    coord.request_stop()
    coord.join(threads)
