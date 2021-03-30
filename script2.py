# global_max = 1
# global_min = 10000000
# lines = open('data/processed_porto_train.csv', 'r').readlines()
# for line in lines:
#     line = eval(line)
#     line_max = max(line)
#     line_min = min(line)
#     if line_max > global_max:
#         global_max = line_max
#     if line_min < global_min:
#         global_min = line_min
#
# lines = open('data/processed_porto_val.csv', 'r').readlines()
# for line in lines:
#     line = eval(line)
#     line_max = max(line)
#     line_min = min(line)
#     if line_max > global_max:
#         global_max = line_max
#     if line_min < global_min:
#         global_min = line_min
#
# print(global_max)
# print(global_min)




import argparse
import tensorflow.compat.v1 as tf

from SDVSAE_t3 import Model
parser = argparse.ArgumentParser()
parser.add_argument('--data_filename', type=str, default="../data/processed_porto{}.csv",
                    help='data file')
parser.add_argument('--map_size', default=[50, 150], type=int, nargs='+',
                    help='size of map')
parser.add_argument('--model_type', type=str, default="sd",
                    help='choose a model')
parser.add_argument('--x_latent_size', type=int, default=32,
                    help='size of input embedding')
parser.add_argument('--rnn_size', type=int, default=256,
                    help='size of RNN hidden state')
parser.add_argument('--mem_num', type=int, default=5,
                    help='size of sd memory')
parser.add_argument('--neg_size', type=int, default=64,
                    help='size of negative sampling')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--model_id', type=str, default="",
                    help='model id')
parser.add_argument('--partial_ratio', type=float, default=1.0,
                    help='partial trajectory evaluation')
parser.add_argument('--eval', type=bool, default=False,
                    help='partial trajectory evaluation')
parser.add_argument('--pt', type=bool, default=False,
                    help='partial trajectory evaluation')
parser.add_argument('--gpu_id', type=str, default="0")
args = parser.parse_args()
model = Model(args)
model_name = "./models/gm_32_256/gm_pretrain"
with tf.Session() as sess:
    model.restore(sess, model_name)

