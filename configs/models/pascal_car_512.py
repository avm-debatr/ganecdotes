from lib.__init__ import *
# Model path - pretrained models are saved in the checkpoints directory by default
# model_path = '/mnt/cloudNAS3/Ankit/1_Datasets/stylegan_pretrained/stylegan2-car-config-f.pt'
model_path = '/mnt/cloudNAS3/Ankit/ganecdotes_expts_2022/e4e_decoder/stylegan2-car-config-f.pt'

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

sample_latents = ROOT_DIR + '/checkpoints/standard/pascal_car_512/latents.pt'
sample_noises  = ROOT_DIR + '/checkpoints/standard/pascal_car_512/noises/'
sample_labels  = ROOT_DIR + '/checkpoints/standard/pascal_car_512/labels.pt'
sample_images  = ROOT_DIR + '/checkpoints/standard/pascal_car_512/images/'

one_shot_ind = 0

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
            'frontside',
            'leftside',
            'rightside',
            'backside',
            'roofside',
            'leftmirror',
            'rightmirror',
            'fliplate',  # front license plate
            'bliplate',  # back license plate
            ] + [f'door_{i}' for i in range(1, 10+1)] \
              + [f'wheel_{i}' for i in range(1, 10+1)] \
              + [f'headlight_{i}' for i in range(1, 10+1)] \
              + [f'window_{i}' for i in range(1, 20+1)]
