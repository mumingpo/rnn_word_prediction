import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
# import numpy as np
import datetime
import json

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


def main():
    '''Train layers and weights with data
    saves state'''

    tf.set_random_seed(0)
    t = ckpt()

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
    x = tf.placeholder(tf.float32, shape=[max_length-S['sl'], S['sl'], charset_size], name='input')
    # x_unpacked = tf.unstack(x, num=max_length-S['sl'], axis=0)
    y_ = tf.placeholder(tf.float32, shape=[max_length-S['sl'], charset_size], name='label')
    cell_stack = tf.nn.rnn_cell.MultiRNNCell([cell(S['hu'], S['do']) for _ in range(S['lc'])])
    t = ckpt(t)
    outputs, state = tf.nn.dynamic_rnn(cell_stack, x, dtype=tf.float32)
    t = ckpt(t)
    bias = tf.Variable(tf.random_normal([max_length-S['sl'], charset_size]))
    pred = layers.fully_connected(outputs[:,4,:], charset_size) + bias
    t = ckpt(t)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=pred))
    optimizer = tf.train.AdagradOptimizer(S['lr'])
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    # saver = tf.train.Saver(pred)
    t = ckpt(t)

    # feed data
    with tf.Session() as sess:
        sess.run(init)
        t = ckpt(t)
        for chap_i, chapter in enumerate(chapters):
            print(chap_i)
            feed = []
            for character in chapter:
                character_one_hot = [0.0]*charset_size
                character_one_hot[c2n[character]] = 1.0
                feed.append(character_one_hot)
            x_feed = []
            y_feed = []
            for i, y in enumerate(feed[:-S['sl']]):
                x_feed.append(feed[i:i+S['sl']])
                y_feed.append(feed[i+S['sl']])
            t = ckpt(t)
            _, total_loss = sess.run([train, loss], feed_dict={
                x: x_feed,
                y_: y_feed
            })
            print(total_loss)
            # saver.save(sess, 'dzz', global_step=chap_i)

if __name__ == '__main__':
    main()
