# Specifications for One-Shot Segmentor
# XS - 3-layer FCN
# S  - 5-layer FCN
# M  - 7-layer FCN
# L  - 9-layer FCN
# Lin - 1-layer Linear MLP

seg_args = dict(size='XS')
# seg_args = dict(size='Lin')

n_layers = 13   # for imsize=512
# n_layers = 13 # for imsize=256

