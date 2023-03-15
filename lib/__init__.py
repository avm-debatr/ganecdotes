import os, sys, time

ROOT_DIR       = os.path.dirname(os.path.dirname(__file__))

BASELINE_DIR   = os.path.join(ROOT_DIR, 'baseline')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
CONFIGS_DIR    = os.path.join(ROOT_DIR, 'configs')
DATA_DIR      = os.path.join(ROOT_DIR, 'data')
LIB_DIR        = os.path.join(ROOT_DIR, 'lib')
MODEL_DIR      = os.path.join(ROOT_DIR, 'models')
SCRIPT_DIR     = os.path.join(ROOT_DIR, 'scripts')
RESULTS_DIR    = os.path.join(ROOT_DIR, 'results')
SRC_DIR        = os.path.join(ROOT_DIR, 'src')
TEST_DIR       = os.path.join(ROOT_DIR, 'tests')

# INC_DIR  = os.path.join(ROOT_DIR, 'include')
# MU_DIR   = os.path.join(INC_DIR, 'mu')
# SPECTRA_DIR = os.path.join(INC_DIR, 'spectra')
# SCANNER_DIR = os.path.join(INC_DIR, 'scanners')
# BAG_DIR = os.path.join(INC_DIR, 'bags')


STYLEGAN_MODELS = os.path.join(CHECKPOINT_DIR, 'standard')
BAGGAN_MODELS   = os.path.join(CHECKPOINT_DIR, 'baggan')

FFHQ_MODEL      = os.path.join(STYLEGAN_MODELS, 'stylegan2-ffhq-config-f.pt')
HUMANS_MODEL    = os.path.join(STYLEGAN_MODELS, 'human_ada.pth')
ANIMALS_MODEL   = os.path.join(STYLEGAN_MODELS, 'animals_ada.pth')


# TO4_DS_DIR = os.path.join(os.path.dirname(ROOT_DIR),
#                                '1_Datasets',
#                                'TO4_ATR_Dataset')

# EXPT_DIR        = os.path.join(os.path.dirname(ROOT_DIR),
#                                'bag_gan_expts')
# MODEL_DIR       = os.path.join(EXPT_DIR, 'models')
# DEFAULT_EXPT    = os.path.join(EXPT_DIR, 'expt_v0.0')
# DEFAULT_MODEL   = os.path.join(EXPT_DIR, 'baggan_v0.0')
# PREPROCESS_DIR  = os.path.join(TO4_DS_DIR, 'preprocess_params')

# DEFAULT_MASK   = os.path.join(INC_DIR, 'samples', 'default_mask.npz')
# DEFAULT_BG     = os.path.join(INC_DIR, 'samples', 'default_background.npz')
# DEFAULT_SAMPLE = os.path.join(INC_DIR, 'samples', 'default_sample.npz')
#
# DEFAULT_DE_BG     = os.path.join(INC_DIR, 'samples', 'default_de_background.npz')
# DEFAULT_DE_SAMPLE = os.path.join(INC_DIR, 'samples', 'default_de_sample.npz')
#
# PIDRAY_SAMPLE_FILE = os.path.join(INC_DIR, 'samples', 'pidray_samples_v1.npz')

