import tensorflow as tf
from tensorflow.contrib import rnn
# from tensorflow.contrib import layers
import numpy as np
import datetime
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

S = {               # settings
    'lc': 3,        # layer count
    'hu': 256,       # num of hidden units
    'lr': 1e-3,     # learning rate
    'do': 0.5,      # drop out rate
    'sl': 12,        # sequence length
}


def ckpt(timestamp=None):
    if timestamp is None:
        return [0, datetime.datetime.now()]
    else:
        now = datetime.datetime.now()
        print('{} seconds elapsed since checkpoint {}'.format((now-timestamp[1]).total_seconds(), timestamp[0]))
        return [timestamp[0]+1, now]


def cell(hidden_units, dropout_rate):
    c = rnn.BasicLSTMCell(hidden_units)
    d = tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=1-dropout_rate)
    return d


def write_chapter(cellstack, n2c, c2n, W, b, m_l, sess):
    state = cellstack.zero_state(1, tf.float32)
    chap = '\x02'
    char = [0.] * len(c2n)
    char[c2n['\x02']] = 1.
    char = tf.convert_to_tensor([char], dtype=tf.float32)
    for i in range(m_l - 1):
        out, state = cellstack(char, state)
        argmaxindex, _ = max(enumerate(sess.run(out)[-1]), key=lambda x: x[1])
        character = n2c[argmaxindex]
        print(character, end='', flush=True)
        chap += character
        char = [0.] * len(c2n)
        char[c2n[character]] = 1.
        char = tf.convert_to_tensor([char], dtype=tf.float32)
    return chap

def main():
    '''Train layers and weights with data
    saves state'''

    tf.set_random_seed(0)
    t = ckpt()
    chaps = []

    # read and data
    with open('readdzz/readdzz/dzz.json', 'r') as f:
        dzz = json.load(f)
    dzz = sorted(dzz, key=lambda chapter: chapter['index'])
    charset = set('\x00\x02\x03')
    max_length = 0
    for chapter in dzz:
        charset |= set(chapter['text'])
        max_length = max(max_length, len(chapter['text']))
    charset_size = len(charset)
    c2n, n2c = dict(), dict()
    for n, c in enumerate(charset):
        n2c[n] = c
        c2n[c] = n
    chapters = []
    for chapter in dzz:
        text = chapter['text']
        text += '\x00' * (max_length-len(text))
        chapters.append('\x02' + text + '\x03')
    max_length += 2
    t = ckpt(t)

    # build graph
    x = tf.placeholder(tf.float32, shape=[1, max_length-1, charset_size], name='input')
    y_ = tf.placeholder(tf.float32, shape=[max_length-1, charset_size], name='label')
    cell_stack = tf.nn.rnn_cell.MultiRNNCell([cell(S['hu'], S['do']) for _ in range(S['lc'])])
    t = ckpt(t)
    outputs, state = tf.nn.dynamic_rnn(cell_stack, x, dtype=tf.float32)
    t = ckpt(t)
    bias = tf.Variable(tf.random_normal([max_length-1, charset_size]))
    W = tf.Variable(tf.random_normal([1, S['hu'], charset_size]))
    pred = tf.matmul(outputs, W)[0] + bias
    t = ckpt(t)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=pred))
    optimizer = tf.train.AdagradOptimizer(S['lr'])
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    #saver = tf.train.Saver(pred)
    t = ckpt(t)

    # feed data
    with tf.Session() as sess:
        sess.run(init)
        t = ckpt(t)
        for chap_i, chapter in enumerate(chapters):
            print(chap_i)

            #if chap_i % 100 == 0:
            #    chaps.append(write_chapter(cell_stack, n2c, c2n, W, bias, max_length, sess))

            feed = []
            for character in chapter:
                character_one_hot = [0.0]*charset_size
                character_one_hot[c2n[character]] = 1.0
                feed.append(character_one_hot)
            x_feed = [feed[1:]]
            y_feed = feed[:-1]
            t = ckpt(t)
            _, loss_ = sess.run([train, loss], feed_dict={
                x: x_feed,
                y_: y_feed
            })
            print(loss_)
            #saver.save(sess, 'dzz', global_step=chap_i)
        chaps.append(write_chapter(cell_stack, n2c, c2n, W, bias, max_length, sess))

    with open('writedzz.json', 'w') as f:
        json.dump(chaps, f)

if __name__ == '__main__':
    main()
