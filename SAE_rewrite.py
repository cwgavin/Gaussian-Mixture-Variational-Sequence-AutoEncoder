import os
import time
import argparse
import tensorflow as tf
from utils import *
from data_generator import DataGenerator
from sklearn.metrics import precision_recall_curve, auc


class Model:
    def __init__(self, args):
        inputs = tf.placeholder(shape=(args.batch_size, None), dtype=tf.int32, name='inputs')
        mask = tf.placeholder(shape=(args.batch_size, None), dtype=tf.float32, name='inputs_mask')
        seq_length = tf.placeholder(shape=args.batch_size, dtype=tf.float32, name='seq_length')

        self.input_form = [inputs, mask, seq_length]

        encoder_inputs = inputs
        decoder_inputs = tf.concat([tf.zeros(shape=(args.batch_size, 1), dtype=tf.int32), inputs], axis=1)
        decoder_targets = tf.concat([inputs, tf.zeros(shape=(args.batch_size, 1), dtype=tf.int32)], axis=1)
        decoder_mask = tf.concat([mask, tf.zeros(shape=(args.batch_size, 1), dtype=tf.float32)], axis=1)

        x_size = out_size = args.map_size[0] * args.map_size[1]
        embeddings = tf.Variable(tf.random.uniform([x_size, args.x_latent_size], -1.0, 1.0), dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

        with tf.name_scope("encoder"):
            encoder_cell = tf.keras.layers.GRUCell(args.rnn_size)
            rnn_input = tf.keras.Input(encoder_inputs_embedded)
            rnn_layer = tf.keras.layers.RNN(encoder_cell, return_state=True)
            encoder_final_state = rnn_layer(rnn_input)[-1]

        fc_w = tf.Variable(tf.random_normal_initializer(stddev=0.02), name="fc_w", shape=[args.rnn_size, args.rnn_size], dtype=tf.float32)
        fc_b = tf.Variable(tf.constant_initializer(0.0), name="fc_b", shape=[args.rnn_size], dtype=tf.float32)

        with tf.name_scope("decoder"):
            decoder_init_state = tf.matmul(encoder_final_state, fc_w) + fc_b
            decoder_cell = tf.keras.layers.GRUCell(args.rnn_size)
            rnn_input = tf.keras.Input(decoder_inputs_embedded)
            rnn_layer = tf.keras.layers.RNN(decoder_cell)
            decoder_outputs = rnn_layer(rnn_input, initial_state=decoder_init_state)

        out_w = tf.Variable(tf.random_normal_initializer(stddev=0.02), name="out_w", shape=[out_size, args.rnn_size], dtype=tf.float32)
        out_b = tf.Variable(tf.constant_initializer(0.0), name="out_b", shape=[out_size], dtype=tf.float32)

        batch_loss = tf.reduce_mean(
            decoder_mask * tf.reshape(
                tf.nn.sampled_softmax_loss(
                    weights=out_w,
                    biases=out_b,
                    labels=tf.reshape(decoder_targets, [-1, 1]),
                    inputs=tf.reshape(decoder_outputs, [-1, args.rnn_size]),
                    num_sampled=args.neg_size,
                    num_classes=out_size
                ), [args.batch_size, -1]
            ), axis=-1
        )
        var_list = [embeddings, fc_w, fc_b, out_w, out_b]
        self.loss = loss = tf.reduce_mean(batch_loss)
        self.train_op = tf.keras.optimizers.Adam(args.learning_rate).minimize(loss, var_list)

        target_out_w = tf.nn.embedding_lookup(out_w, decoder_targets)
        target_out_b = tf.nn.embedding_lookup(out_b, decoder_targets)

        self.batch_likelihood = tf.reduce_mean(
            decoder_mask * tf.math.log_sigmoid(
                tf.reduce_sum(decoder_outputs * target_out_w, -1) + target_out_b
            ), axis=-1, name="batch_likelihood")
        self.batch_encodings = encoder_final_state[0]

        saver = tf.compat.v1.train.Saver(var_list, max_to_keep=10)
        self.save, self.restore = saver.save, saver.restore


def train():
    model = Model(args)
    sampler = DataGenerator(args)

    all_val_loss = []
    # with tf.Session() as sess:
        # tf.global_variables_initializer().run()
    start = time.time()
    for epoch in range(args.num_epochs):
        all_loss = []
        for batch_idx in range(int(sampler.total_traj_num / args.batch_size)):
            batch_data = sampler.next_batch(args.batch_size)
            feed = dict(zip(model.input_form, batch_data))

            loss, _ = sess.run([model.loss, model.train_op], feed)
            all_loss.append(loss)

        val_loss = compute_loss(sess, model, sampler, "val", args)

        if len(all_val_loss) > 0 and val_loss >= all_val_loss[-1]:
            print("Early termination with val loss: {}:".format(val_loss))
            break

        all_val_loss.append(val_loss)

        end = time.time()
        print("epoch: {}\tval loss: {}\telapsed time: {}".format(epoch, val_loss, end - start))
        start = time.time()

        save_model_name = "./models/{}_{}_{}/{}_{}".format(
            args.model_type, args.x_latent_size, args.rnn_size, args.model_type, epoch)
        model.save(save_model_name)


def evaluate():
    model = Model(args)
    sampler = DataGenerator(args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        model_name = "./models/{}_{}_{}/{}_{}".format(
            args.model_type, args.x_latent_size, args.rnn_size, args.model_type, args.model_id)
        model.restore(sess, model_name)

        st = time.time()
        all_likelihood = compute_likelihood(sess, model, sampler, "train")
        elapsed = time.time() - st

        all_prob = np.exp(all_likelihood)

        y_true = np.ones_like(all_prob)
        for idx in sampler.outliers:
            if idx < y_true.shape[0]:
                y_true[idx] = 0

        sd_auc = {}
        sd_index = sampler.sd_index
        for sd, tids in sd_index.items():
            sd_y_true = y_true[tids]
            sd_prob = all_prob[tids]
            if sd_y_true.sum() < len(sd_y_true):
                sd_auc[sd] = auc_score(y_true=sd_y_true, y_score=sd_prob)
        print("Average AUC:", np.mean(list(sd_auc.values())), "Elapsed time:", elapsed)

        sorted_sd_index = sorted(list(sd_auc.keys()), key=lambda k: len(sd_index[k]))
        sorted_sd_auc = [sd_auc[sd] for sd in sorted_sd_index]

        bin_num = 5
        step_size = int(len(sorted_sd_auc) / bin_num)
        for i in range(bin_num):
            print(np.mean(sorted_sd_auc[i*step_size:(i+1)*step_size]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="../data/processed_porto{}.csv",
                        help='data file')
    parser.add_argument('--map_size', default=[50, 150], type=int, nargs='+',
                        help='size of map')
    parser.add_argument('--model_type', type=str, default="s2s",
                        help='choose a model')
    parser.add_argument('--x_latent_size', type=int, default=32,
                        help='size of input embedding')
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--neg_size', type=int, default=64,
                        help='size of negative sampling')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--model_id', type=str, default="SAE",
                        help='model id')
    parser.add_argument('--partial_ratio', type=float, default=1.0,
                        help='partial trajectory evaluation')
    parser.add_argument('--eval', type=bool, default=False,
                        help='partial trajectory evaluation')

    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.eval:
        evaluate()
    else:
        train()
