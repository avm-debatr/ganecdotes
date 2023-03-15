from torch import nn, cat
import torch.nn.functional as F
import torch


class OneShotSegmentor(nn.Module):
    """
    ---------------------------------------------------------------------------
    Architecture borrowed from:    http://repurposegans.github.io/
    ---------------------------------------------------------------------------
    """

    def __init__(self, in_ch, n_class, size='S'):
        super().__init__()

        assert size in ['XS', 'S', 'M', 'L', 'Lin']

        if size=='Lin':
            self.layers = nn.Sequential(nn.Linear(in_ch, n_class),
                                        nn.LeakyReLU(0.2, inplace=True)
                                        )
        else:
            dilations = {
                'XS': [1, 2, 1],
                'S': [1, 2, 1, 2, 1],
                'M': [1, 2, 4, 1, 2, 4, 1],
                'L': [1, 2, 4, 8, 1, 2, 4, 8, 1],
            }[size]

            channels = {
                'XS': [16, 8],
                'S': [128, 64, 64, 32],
                'M': [128, 64, 64, 64, 64, 32],
                'L': [128, 64, 64, 64, 64, 64, 64, 32],
            }[size]
            channels = [in_ch] + channels + [n_class]

            layers = []
            for d, c_in, c_out in zip(dilations, channels[:-1], channels[1:]):
                layers.append(nn.Conv2d(c_in,
                                        c_out,
                                        kernel_size=3,
                                        padding=d,
                                        dilation=d))
                layers.append(nn.LeakyReLU(0.2, inplace=True))

            self.layers = nn.Sequential(*layers[:-1])

        self.channels = n_class
        self.size = size

    def forward(self, x):

        if self.size=='Lin':
            b, c, h, w = x.shape

            x = x.view(b, c, -1).permute(0,2,1)
            x = self.layers(x)
            x = x.permute(0,2,1).reshape(b,self.channels,h,w)

            return x
        else:
            return self.layers(x)


@torch.no_grad()
def concat_features(features, n_layers=13):
    features = features[:n_layers]
    h = max([f.shape[-2] for f in features])
    w = max([f.shape[-1] for f in features])
    return torch.cat([torch.nn.functional.interpolate(f, (h,w), mode='nearest')
                      for f in features], dim=1)

