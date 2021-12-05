import os

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "Li/nlp_learn/datasets/cnn_maildaily_article/chunked/train_*")
eval_data_path = os.path.join(root_dir, "Li/nlp_learn/datasets/cnn_maildaily_article/val.bin")
decode_data_path = os.path.join(root_dir, "Li/nlp_learn/datasets/cnn_maildaily_article/test.bin")
vocab_path = os.path.join(root_dir, "Li/nlp_learn/datasets/cnn_maildaily_article/vocab")
log_root = os.path.join(root_dir, "Li/nlp_learn/pointer_generator_net")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 64
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.4
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 150000

use_gpu=True

lr_coverage=0.15
