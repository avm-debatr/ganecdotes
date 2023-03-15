# Configuration options for training ATP-GAN
from lib.__init__ import *

import os
import time
import numpy as np

###############################################################################
# EXPERIMENT DETAILS

# data locations --------------------------------------------------------------
out_dir = os.path.join(ROOT_DIR,
                       'checkpoints',
                       'baggan',
                       'pidray_baggan_presaved')

baggan_logger_name = 'PIDRay TRAINER'
training_log_path  = os.path.join(out_dir,
                     time.strftime('ganseg_train_%m%d%Y_%H%M%S.log',
                                   time.localtime()))

snap_dir       = os.path.join(out_dir, 'training_snaps')
losses_file    = os.path.join(out_dir, 'training_losses.npz')

net_version    = 'v4.0.1'
checkpoint_dir = os.path.join(out_dir,
                              'models',
                              'expt_%s'%net_version)

# Experiment parameters -------------------------------------------------------

# for training/testing experiment
is_train       = True
ds_type        = 'real'
mode           = 'bagganhq'
test_mode      = None

# preprocessing parameters
image_size      = 256
image_dims      = 384, 384

# training display/save parameters
print_freq       = 400
display_freq     = 2000
losses_to_print  = ['g_gan', 'd',  'g_ppl']
save_by_iter     = False
save_epoch_freq  = 20
save_only_latest = False

train_plot_layout = [5, 5]
# =============================================================================

###############################################################################
# DATASET DETAILS

# dataset loader parameters ---------------------------------------------------

# See BagGAN-HQ repository for more information about these paramaeters
ds_dir=''
subset='train'

batch_size = 20
serial_batches = False
num_threads = 20

# =============================================================================

###############################################################################
# MODEL PARAMETERS

# normalization + layer options
norm         = 'instance'
init_gain    = 0.02
gpu_ids      = [0]
num_channels = 3 # 2

latent_dim   = 512
z_dim, w_dim = latent_dim, latent_dim

generator_params = dict(latent_dims=(z_dim, w_dim),
                        img_resolution=image_size,
                        mlp_layers=8,
                        mlp_lr=0.01,
                        img_chls=num_channels,
                        fir_filter=[1,3,3,1],
                        res2chlmap=None)

disc_params = dict(img_resolution=image_size,
                   img_chls=num_channels,
                   res2chlmap=None,
                   with_q=False)

###############################################################################
# TRAINING PARAMETERS

start_epoch    = 1
n_epochs       = 750

# for continuing/loading saved experiment
continue_train = False # True
load_epoch     = None # 200
load_net       = False # True
verbose        = True

# continue_train = True
# load_epoch     = 500
# load_net       = True
# verbose        = True


# continue_train = True
# load_epoch     = 200
# load_net       = True
# verbose        = True

gan_mode      = 'wgangp' # 'vanilla'  # 'vanilla'

# stylegan2 parameters
use_ppl      = True
r1_lambda = 10             # R1 regularization
ppl_lambda = 2      # weight of the path length regularization
path_batch_shrink = 2 # batch size reducing factor for ppl
ppl_decay  = 0.01
d_reg_every = 16     # interval of applying r1 reg to D
g_reg_every = 4     # interval of applying r1 reg to G
mixing_prob = 0.9   # probability of mixing latent code
chl_multiplier = 2  # channel multiplier
wandb = True
local_rank = 0      #

g_reg_ratio = g_reg_every / (g_reg_every + 1)
d_reg_ratio = d_reg_every / (d_reg_every + 1)

# adaptive discriminator augmentation
augment = True
augment_p = 0
ada_target = 0.6
ada_length = 500*1000
ada_freq = 256

# optimization/loss parameters
lr    = 0.002
beta1 = 0.0

lr_policy      = 'linear'
lr_params = dict(epoch_count=1,
                 n_epochs=100,
                 n_epochs_decay=100,
                 lr_decay_iters=50)

PLOT_TRAINING_LOSS           = True
DISPLAY_TRAINING_OUTPUT      = True

###############################################################################
# VALIDATION PARAMETERS

valid_flag  = True
valid_size  = 100
valid_batch = 10
valid_dir   = os.path.join(out_dir, 'validation')
valid_tests = ['clutter_stats', 'hist_scores', 'hist_plot']
clutter_valid_file = os.path.join(valid_dir,
                                  'clutter_valid_scores.npz')

valid_clutter_range   = [None]*3 # [0.3, 0.5, 0.7]
num_plot_valid_images = 2

VALIDATE_CGAN_PARAM          = False
VALIDATE_HIST_MATCHING_SCORE = True

###############################################################################
# TESTING PARAMETERS

is_train       = False
load_epoch     = 740 # None
load_net       = True # False
max_samples    = None

test_size   = 20
test_batch  = 100
test_dir    = os.path.join(out_dir, 'test')
test_list   = ['clutter_stats', 'hist_scores', 'hist_plot']
clutter_test_file = os.path.join(test_dir, 'clutter_test_scores.npz')

test_clutter_range   = [0.3, 0.35, 0.4, 0.45, 0.5,
                        0.55, 0.6, 0.65, 0.7, 0.75,
                        0.8]
num_plot_test_images = 5000

TEST_CGAN_PARAM          = False
TEST_HIST_MATCHING_SCORE = True

expt_desc = "EXPT. DESCRIPTION: " \
            "BENCHMARK: StyleGAN2 " \
            "Full PIDRay Dataset " \
            "- 512 x 512 res., wgangp loss " \
            "PPL Regularization added + ADA included with random affine"
