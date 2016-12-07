import os
import re
import cPickle as pickle
from collections import defaultdict, Counter

import numpy as np
import tensorflow as tf
from nltk import word_tokenize


def get_spans(fname):
    with open(fname, 'r') as f:
        line_to_spans = defaultdict(list)
        for line in f:
            start, end, phi, tag = line.strip().split('\t')
            start_line, start_token = map(int, start.split(':'))
            end_line, end_token = map(int, end.split(':'))
            assert start_line == end_line

            phi = phi[phi.find('[[') + 2: len(phi) - 2]
            tag = tag[tag.find('[[') + 2: len(tag) - 2]

            span = (start_token, end_token, phi, tag)
            line_to_spans[start_line].append(span)

        return line_to_spans


def tagged_sequences(dir_name):
    text_dir = os.path.join(dir_name, 'txt')
    tags_dir = os.path.join(dir_name, 'tags')
    text_files, tag_files = os.listdir(text_dir), os.listdir(tags_dir)

    text_map = {}
    for tfile in text_files:
        key = os.path.splitext(tfile)[0]
        text_map[key] = tfile

    tag_map = {}
    for tfile in tag_files:
        key = os.path.splitext(tfile)[0]
        tag_map[key] = tfile

    intersection = set(text_map.keys()) & set(tag_map.keys())

    intersection = list(intersection)[:10]

    all_tokens, all_tags = [], []
    for i,key in enumerate(intersection):
        text_file = text_map[key]
        tag_file  =  tag_map[key]
        assert os.path.splitext(text_file)[0] == os.path.splitext(tag_file)[0]

        full_text_fname = os.path.join(text_dir, text_file)
        full_tags_fname = os.path.join(tags_dir, tag_file)

        line_to_spans = get_spans(full_tags_fname)
        with open(full_text_fname, 'r') as text_file:
            for line_num, line in enumerate(text_file):
                if line.strip():
                    spans = line_to_spans[line_num]
                    tokens = word_tokenize(line.strip())

                    index, tags = 0, []
                    for span in spans:
                        tags.extend(['OUTSIDE']*(span[0] - index))
                        tags.append(span[3] + '-B')
                        tags.extend([span[3] + '-I']*(span[1] - span[0]))
                        index = span[1] + 1
                    tags.extend(['OUTSIDE']*(len(tokens) - len(tags)))

                    tokens = process(tokens)
                    all_tokens.append(tokens)
                    all_tags.append(tags)

    return all_tokens, all_tags


def load_embeddings(embedding_path):
    with open(embedding_path, 'r') as embedding_file:
        W_idx, W = [], []
        for line_num, line in enumerate(embedding_file):
            items = line.split()
            W_idx.append(items[0])
            W.append(items[1:])
        return np.array(W_idx), np.array(W)


def process(sequence):
    new_sequence = []
    for token in sequence:
        token = re.sub(r'[0-9]+', '<NUM>', token)
        token = token.lower()
        new_sequence.append(token)
    return new_sequence


def get_vocab(train_sequences, test_sequences, W_idx, min_count=3):
    train_tokens = [token for seq in train_sequences for token in seq]
    test_tokens = [token for seq in test_sequences for token in seq]
    train_counts = Counter(train_tokens)

    remove = set(t for t in train_counts if train_counts[t] <= min_count)
    keep = set(train_counts.keys()) - remove
    emb = set(W_idx)
    train = set(train_tokens)
    test = set(test_tokens)

    return (keep - emb) | (((train | test) & emb) - remove)


def encode(sequences, index):
    encoded_sequences = []
    for sequence in sequences:
        encoded_sequence = []
        for item in sequence:
            encoded_sequence.append(index.get(item, 1))
        encoded_sequences.append(encoded_sequence)
    return encoded_sequences


def create_embedding_matrix(W_idx, W, index):
    emb_words = set(W_idx)
    new_words, new_embeddings = [], []
    for word in index:
        if word not in emb_words:
            new_words.append(word)
            new_embeddings.append(np.random.normal(size=W.shape[1]))

    W_idx = np.append(W_idx, np.array(new_words))
    W = np.vstack((W, np.array(new_embeddings)))

    W_idx = np.array([index[i] if i in index else np.inf for i in W_idx])
    W = W[np.argsort(W_idx)][:len(index), :]
    return W.astype(np.float16)


def make_example(tokens, tags):
    ex = tf.train.SequenceExample()
    sequence_length = len(tokens)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["tags"]
    for token, tag in zip(tokens, tags):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(tag)
    return ex


def to_tfrecords(sequences, tag_sequences, output_path):
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for tokens, tags in zip(sequences, tag_sequences):
            example = make_example(tokens, tags)
            writer.write(example.SerializeToString())


def create_index(vocab):
    full_vocab = ['<PAD>', '<UNK>'] + list(vocab)
    return dict(map(reversed, enumerate(full_vocab)))


def create_records():
    W_idx, W = load_embeddings('embeddings/glove.6B.50d.txt')
    train_sequences, train_tags = tagged_sequences('../data/train')
    test_sequences, test_tags = tagged_sequences('../data/test')

    tag_vocab = set([tag for seq in train_tags for tag in seq])
    token_vocab = get_vocab(train_sequences, test_sequences, W_idx)
    word_index, tag_index = create_index(token_vocab), create_index(tag_vocab)

    train_sequences = encode(train_sequences, word_index)
    train_tags = encode(train_tags, tag_index)
    test_sequences = encode(test_sequences, word_index)
    test_tags = encode(test_tags, tag_index)

    W = create_embedding_matrix(W_idx, W, word_index)
    np.savetxt('embeddings/embeddings.txt', W, fmt='%.6f')
    pickle.dump(word_index, open('indexes/word_index.pkl', 'w'))
    pickle.dump(tag_index, open('indexes/tag_index.pkl', 'w'))
    to_tfrecords(train_sequences, train_tags, 'tfrecords/train.tfrecords')
    to_tfrecords(test_sequences, test_tags, 'tfrecords/test.tfrecords')


if __name__ == '__main__':
    create_records()
