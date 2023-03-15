from lib.__init__ import *
# Model path - pretrained models are saved in the checkpoints directory by default
# model_path = '/mnt/cloudNAS3/Ankit/1_Datasets/stylegan_pretrained/stylegan2-horse-config-f.pt'
model_path = '/mnt/cloudNAS3/Ankit/ganecdotes_expts_2022/e4e_decoder/stylegan2-horse-config-f.pt'

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

sample_latents = ROOT_DIR + '/checkpoints/standard/pascal_horse_256/latents.pt'
sample_noises  = ROOT_DIR + '/checkpoints/standard/pascal_horse_256/noises/'
sample_labels  = ROOT_DIR + '/checkpoints/standard/pascal_horse_256/labels.pt'
sample_images  = ROOT_DIR + '/checkpoints/standard/pascal_horse_256/images/'

one_shot_ind = 11

# classes = ['background',
#            'skin',
#            'hair',
#            'eye',
#            'eyebrow',
#            'nose',
#            'mouth',
#            'ear']

# classes = [ 'background',   # 0
#             'skin',         # 1
#             'neck',         # 2
#             'hat',          # 3
#             'eye_g',        # 4
#             'hair',         # 5
#             'ear_r',        # 6
#             'neck_l',       # 7
#             'cloth',        # 8
#             'l_eye',        # 9
#             'r_eye',        # 10
#             'l_brow',       # 11
#             'r_brow',       # 12
#             'nose',         # 13
#             'l_ear',        # 14
#             'r_ear',        # 15
#             'mouth',        # 16
#             'u_lip',        # 17
#             'l_lip']        # 18

classes = ['background',
            'head',  # 1
            'leye',  # left eye 2
            'reye',  # right eye 3
            'lear',  # left ear 4
            'rear',  # right ear 5
            'muzzle', # 6
            'lhorn',  # left horn 7
             'rhorn',  # right horn 8
             'torso',  # 9
             'neck',   # 10
             'lfuleg',  # left front upper leg 11
             'lflleg',  # left front lower leg 12
             'rfuleg',  # right front upper leg 13
             'rflleg',  # right front lower leg 14
             'lbuleg',  # left back upper leg 15
             'lblleg',  # left back lower leg 16
             'rbuleg',  # right back upper leg 17
             'rblleg',  # right back lower leg 18
             'tail'] + \
          ['20n','21n','22n',   # no annotations for labels from 20-29
           '23n','24n','25n',
           '26n','27n','28n',
           '29'] + \
           ['lfho', # 30
            'rfho',
            'lbho',
            'rbho']
