import tensorflow as tf
import data_wrapper

batch_size = 20
hidden_units = 32
learning_rate = .005
training_steps = 10**5

graph = tf.Graph()
reader = data_wrapper.DataReader()
output_units = len(reader.tag_index)

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
    z = tf.reshape(z, [batch_size, -1, output_units])

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

            # print some info about the batch
            print 'Loss:    ', batch_loss
            print 'Accuracy:', accuracy
            print 'Sentence:', reader.decode_tokens(x[0][(y != 0)[0]][:15])
            print 'Truth:   ', reader.decode_tags(y[0][(y != 0)[0]][:15])
            print 'Pred:    ', reader.decode_tags(y_[0][(y != 0)[0]][:15])
            print

    coord.request_stop()
    coord.join(threads)
