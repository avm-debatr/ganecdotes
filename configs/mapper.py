from lib.util.util import *
import torch.optim.lr_scheduler as sch
import torch.nn as nn


# Maps config files for different models and segmentors

# StyleGAN Models
models = {'ffhq-256':     os.path.join(CONFIGS_DIR, 'models', 'ffhq_256.py'),
          'ffhq-256-er':  os.path.join(CONFIGS_DIR, 'models', 'ffhq_256_rp_earr.py'),
          'ffhq-256-eg':  os.path.join(CONFIGS_DIR, 'models', 'ffhq_256_rp_eyeg.py'),
          'car-512':      os.path.join(CONFIGS_DIR, 'models', 'lsun_car_512.py'),
          'cat-256':      os.path.join(CONFIGS_DIR, 'models', 'lsun_cat_256.py'),
          'horse-256':    os.path.join(CONFIGS_DIR, 'models', 'lsun_horse_256.py'),
          'horse-256-rp': os.path.join(CONFIGS_DIR, 'models', 'lsun_horse_256_rp.py'),
          'church-256':   os.path.join(CONFIGS_DIR, 'models', 'lsun_church_256.py'),
          'church-512':   os.path.join(CONFIGS_DIR, 'models', 'lsun_church_512.py'),
          'pidray-256':   os.path.join(CONFIGS_DIR, 'models', 'pidray_bag_256.py'),
          'pidray-pliers-256': os.path.join(CONFIGS_DIR, 'models', 'pidray_pliers_256.py'),
          'pidray-hammer-256': os.path.join(CONFIGS_DIR, 'models', 'pidray_hammer_256.py'),
          'pidray-powerbank-256': os.path.join(CONFIGS_DIR, 'models', 'pidray_powerbank_256.py'),
          'pidray-wrench-256': os.path.join(CONFIGS_DIR, 'models', 'pidray_wrench_256.py'),
          'pidray-handcuffs-256': os.path.join(CONFIGS_DIR, 'models', 'pidray_handcuffs_256.py'),
          'pidray-256':   os.path.join(CONFIGS_DIR, 'models', 'pidray_bag_256.py'),
          'celeba-256':   os.path.join(CONFIGS_DIR, 'models', 'celebamask_ffhq_im_256_n_100.py'),
          'p-horse-256':  os.path.join(CONFIGS_DIR, 'models', 'pascal_horse_256.py'),
          'p-car-512':    os.path.join(CONFIGS_DIR, 'models', 'pascal_car_512.py'),
          'afhq-256':     os.path.join(CONFIGS_DIR, 'models', 'afhq_256.py'),
          }

# Segmentor types - contains hfc_with_swav networks + baselines
segmentors = {'repurposegan':        os.path.join(CONFIGS_DIR, 'segmentors', 'repurposegan_config.py'),
              'datasetgan':          os.path.join(CONFIGS_DIR, 'segmentors', 'datasetgan_config.py'),
              'hfc_with_swav':       os.path.join(CONFIGS_DIR, 'segmentors', 'hfc_with_swav_config.py'),
              'hfc_with_simclr':     os.path.join(CONFIGS_DIR, 'segmentors', 'hfc_with_simclr_config.py'),
              'hfc_kmeans':          os.path.join(CONFIGS_DIR, 'segmentors', 'hfc_kmeans_config.py'),
              'hfc_with_swav_cat':   os.path.join(CONFIGS_DIR, 'segmentors', 'hfc_with_swav_cat_config.py'),
              'hfc_with_swav_car':   os.path.join(CONFIGS_DIR, 'segmentors', 'hfc_with_swav_car_config.py'),
              'hfc_with_swav_ffhq':  os.path.join(CONFIGS_DIR, 'segmentors', 'hfc_with_swav_ffhq_config.py'),
              'hfc_with_swav_horse': os.path.join(CONFIGS_DIR, 'segmentors', 'hfc_with_swav_horse_config.py'),
              'hfc_with_swav_pidray': os.path.join(CONFIGS_DIR, 'segmentors', 'hfc_with_swav_pidray_config.py'),
              }

# training method - normal, hfc_kmeans, hfc_with_swav, lagm
trainer = {'supervised':    os.path.join(CONFIGS_DIR, 'trainers', 'supervised_config.py'),
          }

# tester module - not used
tester = {'iou':        os.path.join(CONFIGS_DIR, 'tester', 'iou_config.py'),
          'roc':        os.path.join(CONFIGS_DIR, 'tester', 'roc_config.py'),
          'prcurve':    os.path.join(CONFIGS_DIR, 'tester', 'prcurve_config.py'),
          'dice':       os.path.join(CONFIGS_DIR, 'tester', 'dice_config.py'),
          'conf_mat':   os.path.join(CONFIGS_DIR, 'tester', 'conf_mat_config.py'),
          'all':        os.path.join(CONFIGS_DIR, 'tester', 'all_config.py')
         }

losses = {'bce':            nn.BCEWithLogitsLoss,
          'softmax':        nn.Softmax,
          'sigmoid':        nn.Sigmoid,
          'tanh':           nn.Tanh,
          'logloss':        nn.LogSoftmax,
          'cross_entropy':  nn.CrossEntropyLoss
         }

lr_scheduler = {'step':     sch.StepLR,
                'plateau':  sch.ReduceLROnPlateau,
                'cosine':   sch.CosineAnnealingLR
               }