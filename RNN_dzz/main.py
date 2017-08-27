import tensorflow as tf
from tensorflow.contrib import rnn
import json
import datetime

SETTINGS = {
    'rnn_layer_units_count': 64,
    'learning_rate': 1e-3,
    'training_dropout_rate': 0.5,
    'layers': 1,
    'segment_length': 9
}
timestamp = [0, datetime.datetime.now()]


def lstm_cell_with_dropout(units, dropout_rate):
    return rnn.DropoutWrapper(rnn.BasicLSTMCell(units),
                              output_keep_prob=1-dropout_rate)


def ckpt():
    global timestamp
    now = datetime.datetime.now()
    print('{} seconds elapsed since checkpoint {}'.format((now-timestamp[1]).total_seconds(), timestamp[0]))
    timestamp = [timestamp[0] + 1, now]


def main():
    '''what is rnn? what is machine learning? what is a computer? help!'''

    # set random state for reproducibility
    tf.set_random_seed(0)

    # read training data
    with open(r'readdzz/readdzz/dzz.json', 'r') as f:
        dzz = json.load(f)
    dzz = sorted(dzz, key=lambda chap: chap['index'])
    charset = set('\x00')
    max_length = 0
    num_of_chapters = 0
    for chapter in dzz:
        charset |= set(chapter['text'])
        max_length = max(max_length, len(chapter['text']))
        num_of_chapters += 1
    charset_size = len(charset)
    charset = list(charset)
    c2n = dict()
    n2c = dict()
    for n, c in enumerate(charset):
        n2c[n] = c
        c2n[c] = n
    ckpt()#1
    # define graph
    text_input = tf.placeholder(tf.float32,
                                [max_length-SETTINGS['segment_length'],
                                 SETTINGS['segment_length'] ,
                                 charset_size],
                                'input')
    x = tf.unstack(text_input, num=max_length-SETTINGS['segment_length'], axis=0)
    y_ = tf.placeholder(tf.float32, [None, charset_size], 'label')
    W = tf.Variable(tf.random_normal([SETTINGS['rnn_layer_units_count'], charset_size]))
    b = tf.Variable(tf.random_normal([charset_size]))
    ckpt()#2
    cell = lstm_cell_with_dropout(SETTINGS['rnn_layer_units_count'], SETTINGS['training_dropout_rate'])
    ckpt()#3
    cellstack = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_with_dropout(SETTINGS['rnn_layer_units_count'],
                                                                    SETTINGS['training_dropout_rate'])
                                             for _ in range(SETTINGS['layers'])])
    ckpt()#4
    output, state = rnn.static_rnn(cellstack, x, dtype=tf.float32)
    ckpt()#5
    y = tf.matmul(output[-1], W) + b
    ckpt()#6
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    ckpt()#7
    optimizer = tf.train.AdagradOptimizer(SETTINGS['learning_rate'])
    ckpt()#8
    train = optimizer.minimize(loss)
    ckpt()#9
    init = tf.global_variables_initializer()
    ckpt()#10

    # feed data
    with tf.Session() as sess:
        ckpt()#11
        sess.run(init)
        ckpt()#12
        for chapter in dzz:
            print(chapter['index'])
            text_onehot = []
            feed = []
            for c in chapter['text']:
                c_onehot = [0.] * charset_size
                c_onehot[c2n[c]] = 1.
                text_onehot.append(c_onehot)
            filler = [0.] * charset_size
            filler[c2n['\x00']] = 1.
            filler_length = max_length - len(chapter['text'])
            for _ in range(filler_length):
                text_onehot += filler.copy()
            for i in range(int(max_length - SETTINGS['segment_length'] - 1)):
                feed.append(text_onehot[i:i+SETTINGS['segment_length']])
            sess.run(train, feed_dict={text_input: feed, y_: text_onehot[SETTINGS['segment_length']:]})
            print(sess.run(loss))

if __name__ == '__main__':
    main()