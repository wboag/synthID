import functools
import sets
import tensorflow as tf
import data_wrapper

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class VariableSequenceLabelling:

    def __init__(self, num_hidden=256, num_layers=2):
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize
        self.inputs
        self.tags
        self.lengths


    @lazy_property
    def prediction(self):

        # read batch and look up embeddings
        tokens, tags, lengths = reader.get_train_batch(batch_size)
        inputs = tf.nn.embedding_lookup(tf.Variable(reader.embeddings), tokens)

        print inputs, inputs.get_shape()[1]
        print tags, tags.get_shape()
        print lengths, lengths.get_shape()

        # Recurrent network.
        output, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self._num_hidden)]*self._num_layers),
            inputs,
            dtype=tf.float32,
            sequence_length=lengths
        )
        tf.get_variable_scope().reuse_variables()
        
        # last layer weights
        max_length = inputs.get_shape()[1]
        num_classes = output_units
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        
        # flatten + softmax
        output = tf.reshape(output, [-1, self._num_hidden])
        logits = tf.matmul(output, weight) + bias
        prediction = tf.nn.softmax(logits)
        
        # reshape
        logits = tf.reshape(logits, [-1, max_length, num_classes])
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return logits, prediction

 
    @lazy_property
    def cross_ent(self):
        predictions = self.prediction[0]
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(predictions, self.tags)
        return ce
        

    @lazy_property
    def optimize(self):
        learning_rate = 0.0003
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.ce)

    @lazy_property
    def error(self):
        rounded = tf.round(self.prediction[1])
        mistakes = tf.not_equal(self.tags, rounded)
        mistakes = tf.cast(mistakes, tf.float32)
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        lengths = tf.expand_dims(tf.cast(self.lengths, tf.float32), 1)
        mistakes /= self.length
        return tf.reduce_mean(mistakes)

        
    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)
    

config = tf.ConfigProto(allow_soft_placement = True)
with tf.Session(config = config) as sess, tf.device('/cpu'):
    reader = data_wrapper.DataReader()
    batch_size = 20
    output_units = len(reader.tag_index)
    model = VariableSequenceLabelling()
    sess.run(tf.initialize_all_variables())
    for step_num in range(training_steps):
        _, cost, prediction, error = sess.run([model.optimize, model.cross_ent, model.prediction, model.error])
        print('Step {} : Cost = {}, Error = {}'.format(step_num, cost, error))
            