from torch import nn, cat
import torch.nn.functional as F
import torch


class PixelClassifier(nn.Module):
    """
    classifier model from DatasetGAN repo:
    https://github.com/nv-tlabs/datasetGAN_release.git

    """
    def __init__(self, in_ch, n_class, size='S'):
        super(PixelClassifier, self).__init__()
        if n_class < 32:
            self.layers = nn.Sequential(
                nn.Linear(in_ch, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, n_class),
                # nn.Sigmoid()
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_ch, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, n_class),
                # nn.Sigmoid()
            )
        self.n_class = n_class

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
        blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and \
                    (classname.find('Conv') != -1
                        or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).flatten(0, 2)
        x = self.layers(x)
        return x.reshape(b,h,w,self.n_class).permute(0, 3, 1,2)
# -----------------------------------------------------------------------------


@torch.no_grad()
def concat_features(features, n_layers=13):
    """
    ---------------------------------------------------------------------------
    creates a single feature vector for each pixel from the hidden features
    :param features:
    :param n_layers:
    :return:
    ---------------------------------------------------------------------------
    """
    features = features[:n_layers]
    h = max([f.shape[-2] for f in features])
    w = max([f.shape[-1] for f in features])
    return torch.cat([torch.nn.functional.interpolate(f, (h,w),
                                                      mode='nearest')
                      for f in features], dim=1)
# -----------------------------------------------------------------------------
