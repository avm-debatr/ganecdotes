from lib.__init__ import *
# Model path - pretrained models are saved in the checkpoints directory by default
model_path = ROOT_DIR + '/checkpoints/standard/lsun_cat_256/stylegan2-cat-config-f.pt'

# Default parameters for pretrained model
num_latents_for_mean = 4096
truncation = 0.7
image_size = 256
latent_dim = 512

gen_args = dict(size=image_size,
                style_dim=latent_dim,
                n_mlp=8)

# config path required when using Baggage Datasets - PIDRay, SIXray, GDXray
# config path = ''
is_baggan = False

# Ensure that the data is downloaded from link in repo to the checkpoints directory
sample_latents = ROOT_DIR + '/checkpoints/standard/lsun_cat_256/latents.pt'
sample_images = ROOT_DIR + '/checkpoints/standard/lsun_cat_256/images/'
sample_labels = ROOT_DIR + '/checkpoints/standard/lsun_cat_256/labels.pt'

# Set this to the image index to be used for one-shot learning
one_shot_ind = 0

# label index = list index
classes = ['background',
           'head',
           'eye',
           'nose',
           'ear',
           'neck',
           'torso',
           'tail',
           'f_leg',
           'b_leg']
