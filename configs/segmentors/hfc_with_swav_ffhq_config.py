n_layers = 13
n_hfc_layers = 6


train_hfc = True
layer_hf_dim = [512, 1024, 1024, 1024, 1024, 512, 256]
hlen =   sum(layer_hf_dim) # 4864
nclasses = 512

hfc_prep_args = dict(
    # for image augmentation ==================================================
    perturb_args=dict(truncation=0.7,
                      n_layers=n_hfc_layers,
                      n_samples=1,
                      layer_no=None,
                      perturb_std=[1.0]*n_hfc_layers),

    # for swav clustering =====================================================
    swav_args=dict(
                   # ----------------------------------------------------------
                   # Number of Training epochs
                   num_epochs=100,
                   num_samples=1,       # samples per epoch
                   # No. of patches taken from the same image
                   # Each patch will be subjected to a different transform
                   num_patches=5,
                   # Size to crop from image
                   sampling_method='random',  # whether to sample with patches
                                              # or random pixels {patch|random}
                   patch_size=20000,           # nclasses*80,  # 64*64,
                   hf_interp='nearest',
                   # ----------------------------------------------------------
                   # Optimizer Parameters
                   warmup_epochs=100,           # Warmup epochs with high lr
                   start_warmup=0.01,           # Starting lr

                   # LR Scheduler - cosine
                   use_scheduler=False,         # Flag for lr scheduling
                   base_lr=0.01,                # starting lr
                   final_lr=0.0001,             # ending lr
                   trust_coeff=0.01,            # LARC trust coeff
                   freeze_prototype_niters=313, # Not used

                   # SGD Optimizer
                   train_args=dict(lr=0.01,
                                   momentum=0.9),
                   # ----------------------------------------------------------
                   # SwAV Loss
                   projn_nw='linear',           # projection n/w
                                                # {linear | 1-layer | 2-layer}
                   temperature=0.01,            # temperature for CE Loss
                   nprototypes=5000,            # Prototype vectors
                   nclasses=nclasses,           # Proj. Feature Length
                   hlen=hlen,                   # no. of hidden features
                   add_local_loss=False,

                   # display
                   plot_test_images=False,
                   epoch_print_freq=5,
                   max_masks=4),

    # For Sinkhorn-Knopp algorithm ============================================
    sinkhorn_args=dict(source_pdf='uniform',
                       niters=10,
                       eps=0.005),
    train=train_hfc,
    layer_hf_dim = [512, 1024, 1024, 1024, 1024, 512, 256]
)
# =============================================================================

# Specifications for One-Shot Segmentor
# XS - 3-layer FCN
# S  - 5-layer FCN
# M  - 7-layer FCN
# L  - 9-layer FCN

seg_args = dict(size='XXS',
                in_ch=nclasses)
