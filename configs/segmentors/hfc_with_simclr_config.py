n_layers = 13   # for imsize=512
n_hfc_layers = 6

train_hfc = True
layer_hf_dim = [512, 1024, 1024, 1024, 1024, 512, 256]
hlen = sum(layer_hf_dim) # 5376
nclasses = 512

hfc_prep_args = dict(
    # for image augmentation ==================================================
    perturb_args=dict(truncation=0.7,
                      n_layers=n_hfc_layers,
                      n_samples=1,
                      layer_no=None,
                      perturb_std=[1.0]*n_hfc_layers),

    # for simclr clustering =====================================================
    simclr_args=dict(
                   # ----------------------------------------------------------
                   # Number of Training epochs
                   num_iters=100,
                   # No. of patches taken from the same image
                   # Each patch will be subjected to a different transform
                   batch_size=20,
                   # Size to crop from image
                   patch_size=20000,
                   hf_interp='nearest',
                   # ----------------------------------------------------------
                   # Optimizer Parameters
                   trust_coeff=0.01,

                   # SGD Optimizer
                   train_args=dict(lr=0.01,
                                   momentum=0.9),
                   # Adam
                   # ----------------------------------------------------------
                   # SimCLR Loss
                   temperature=1.0,            # temperature for CE Loss
                   nclasses=nclasses,           # Proj. Feature Length
                   hlen=hlen,                   # no. of hidden features
                   epoch_print_freq=5,
                   max_masks=4),

    train=train_hfc,
    layer_hf_dim = layer_hf_dim
)
# =============================================================================

# Specifications for One-Shot Segmentor
# XS - 3-layer FCN
# S  - 5-layer FCN
# M  - 7-layer FCN
# L  - 9-layer FCN

seg_args = dict(size='XS',
                in_ch=nclasses)
