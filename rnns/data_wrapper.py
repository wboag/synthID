import numpy as np
import tensorflow as tf
import cPickle as pickle


class DataReader(object):

    def __init__(self):
        embedding_filename = '../preprocessing/embeddings/embeddings.txt'
        word_index_filename = '../preprocessing/indexes/word_index.pkl'
        tag_index_filename = '../preprocessing/indexes/tag_index.pkl'

        self.embeddings = np.loadtxt(embedding_filename, dtype=np.float32)
        self.word_index = pickle.load(open(word_index_filename, 'r'))
        self.tag_index = pickle.load(open(tag_index_filename, 'r'))
        self.word_index_inv = dict(map(reversed, self.word_index.items()))
        self.tag_index_inv = dict(map(reversed, self.tag_index.items()))

    def encode_tokens(self, tokens):
        return map(self.word_index.get, tokens)

    def decode_tokens(self, encoded_tokens):
        return ' '.join(map(self.word_index_inv.get, encoded_tokens))

    def encode_tags(self, tags):
        return map(self.tag_index.get, tags)

    def decode_tags(self, encoded_tags):
        return ' '.join(map(self.tag_index_inv.get, encoded_tags))

    def embeddings_from_tokens(self, tokens):
        return self.embeddings[np.array(self.encode_tokens(tokens))]

    def embeddings_from_encoded_tokens(self, encoded_tokens):
        return self.embeddings[np.array(encoded_tokens)]

    def get_train_batch(self, batch_size):
        fname = '../preprocessing/tfrecords/train.tfrecords'
        return self.get_batch(batch_size, fname, 'train')

    def get_test_batch(self, batch_size):
        fname = '../preprocessing/tfrecords/test.tfrecords'
        return self.get_batch(batch_size, fname, 'test')

    def get_batch(self, batch_size, fname, mode):
        shuffle = mode == 'train'
        num_epochs = None if mode == 'train' else 1
        allow_smaller_final_batch = mode == 'test'

        queue = tf.train.string_input_producer([fname], num_epochs, shuffle)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)

        context_feats = {"length": tf.FixedLenFeature([], dtype=tf.int64)}
        fixed_len_seq = tf.FixedLenSequenceFeature([], dtype=tf.int64)
        seq_feats = {"tokens": fixed_len_seq, "tags": fixed_len_seq}

        context, seq = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_feats,
            sequence_features=seq_feats
        )
        tokens, tags, length = tf.train.batch(
            tensors=[seq['tokens'], seq['tags'], context['length']],
            allow_smaller_final_batch=allow_smaller_final_batch,
            batch_size=batch_size,
            dynamic_pad=True,
            capacity=1000
        )
        return tokens, tags, length
