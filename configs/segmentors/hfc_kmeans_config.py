n_layers = 13   # for imsize=512
n_hfc_layers = 5

# Parameters for KMeans clustering for HFC

clusters_per_layer = [4, 8, 16, 32, 64]     # define no. of clusters for each
                                            # StyleGAN layer - typically in
                                            # powers  of 2
train_hfc = True                            # if to be trained from sratch or
                                            # use pre-saved model
# -----------------------------------------------------------------------------
hfc_prep_args = dict(

    # params for latent vector perturbation
    perturb_args=dict(truncation=0.7,
                      n_layers=n_hfc_layers,
                      n_samples=4,
                      perturb_std=[1.0]*n_hfc_layers
                      ),

    # whetehr flat or hierarchical clustering is to be used
    hfc_algo='hfc_kmeans',
    hfc_args=dict(
        # sklearn.cluster.kmeans arguments
        kmeans_args=dict(# n_init=1,
                         # max_iter=1000,
                         verbose=0),
        # params for preprocessor
        base_args=dict(out_dir=None, # if you need a separate dir
                       n_layers=n_hfc_layers,
                       clusters_per_layer=clusters_per_layer,
                       out_size=256,    # output image size
                       presaved=not train_hfc # if using presaved Kmeans models
                       )
    ),
        hier_encode=False,      # use bayesian belief to cualate prob. scores
        hle_samples=100,        # samples for belief calculation
        train=train_hfc
)

# -----------------------------------------------------------------------------
# For using gaussian mixture models instead of KMeans models

# hfc_prep_args = dict(perturb_args=dict(truncation=0.7,
#                                        n_layers=n_hfc_layers,
#                                        n_samples=5,
#                                        perturb_std=[1.0]*n_hfc_layers),
#                      hfc_algo='gmm',
#                      hfc_args=dict(gmm_args=dict(n_init=5,
#                                                  max_iter=100,
#                                                  reg_covar=1e-6,
#                                                  tol=1e-3,
#                                                  verbose=0),
#                                    base_args=dict(out_dir=None,
#                                                   n_layers=n_hfc_layers,
#                                        clusters_per_layer=clusters_per_layer,
#                                                   out_size=256,
#                                                   presaved=False)),
#                      hier_encode=False,
#                      train=True)
# -----------------------------------------------------------------------------

# Specifications for One-Shot Segmentor
# S  - 5-layer FCN
# M  - 7-layer FCN
# L  - 9-layer FCN

seg_args = dict(size='S',
                in_ch=sum(clusters_per_layer))
