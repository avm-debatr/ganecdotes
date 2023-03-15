from lib.__init__ import *
# Model path - pretrained models are saved in the checkpoints directory by default
model_path = '/mnt/cloudNAS3/Ankit/1_Datasets/stylegan_pretrained/stylegan2-church-config-f.pt'

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

sample_latents = ROOT_DIR + '/checkpoints/standard/lsun_church_256/latents.pt'
sample_images = ROOT_DIR + '/checkpoints/standard/lsun_church_256/images/'
sample_labels = ROOT_DIR + '/checkpoints/standard/lsun_church_256/labels.pt'

one_shot_ind = 0

classes = ['background', 'roof',  'window', 'door', 'steps',
           'turret',    'wall', 'road'
          ]
