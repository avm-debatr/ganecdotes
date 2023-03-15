from lib.__init__ import *

# Contains BaGGAN related parameters - do not change
config_path = MODEL_DIR + '/baggan/config/config_pidray_unlabeled.py'

# Default parameters for pretrained model
num_latents_for_mean = 4096
truncation = 0.95
image_size = 256
latent_dim = 512


gen_args = dict(size=256,
                style_dim=512,
                n_mlp=8,
                channel_multiplier=2,
                blur_kernel=[1, 3, 3, 1],
                lr_mlp=0.01)

is_baggan = True

# Ensure that the data is downloaded from link in repo to the checkpoints directory
sample_latents = ROOT_DIR + '/checkpoints/baggan/pidray_powerbank_256/latents.pt'
sample_images = ROOT_DIR + '/checkpoints/baggan/pidray_powerbank_256/images/'
sample_labels = ROOT_DIR + '/checkpoints/baggan/pidray_powerbank_256/labels.pt'

# Set this to the image index to be used for one-shot learning
one_shot_ind = 19

# label index = list index
classes = ['background', 'powerbank']
