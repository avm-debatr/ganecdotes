import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from numpy import *
import random
from skimage.exposure import match_histograms, histogram_matching
import torch.nn.functional as F
# from .one_hot import one_hot


class Identity(nn.Module):
    """
    ---------------------------------------------------------------------------
    Class Description:
        Identity network - simply forwards the input values.
    ---------------------------------------------------------------------------
    """

    def forward(self, x):
        """
        -----------------------------------------------------------------------
        forward function.

        :param x:   input value
        :return:
        -----------------------------------------------------------------------
        """
        return x
# -----------------------------------------------------------------------------


def get_norm_layer(norm_type='instance'):
    """
    ---------------------------------------------------------------------------
    Return a normalization layer

    :param norm_type - the name of the normalization layer:
                             {'batch' | 'instance' | 'none'}

    'batch'     - use learnable affine parameters and track running statistics
                  (mean/stddev)
    'instance'  - affine parameters/tracking statistics not used
    'none'      - no normalization of layers.

    ---------------------------------------------------------------------------
    """

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d,
                                       affine=True,
                                       track_running_stats=True)
    elif norm_type == 'instance':
        # norm_layer = functools.partial(nn.InstanceNorm2d,
        #                                affine=False,
        #                                track_running_stats=False)
        norm_layer = functools.partial(nn.InstanceNorm2d,
                                       affine=True,
                                       track_running_stats=True)

    elif norm_type == 'none':
        def norm_layer(x): return Identity()

    else:
        raise NotImplementedError('normalization layer [%s] is not '
                                  'found' % norm_type)
    return norm_layer
# -----------------------------------------------------------------------------


def get_scheduler(optimizer,
                  lr_policy,
                  epoch_count=None,
                  n_epochs=None,
                  n_epochs_decay=None,
                  lr_decay_iters=None):
    """
    ---------------------------------------------------------------------------
    Return a learning rate scheduler

    :param optimizer          the optimizer of the network
    :param opt (option class) stores all the experiment flags; needs to be a
                              subclass of BaseOptions．opt.lr_policy is the name
                              of learning rate policy:
                              linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs>
    epochs and linearly decay the rate to zero over the next <opt.n_epochs_decay>
    epochs. For other schedulers (step, plateau, and cosine), we use the default
    PyTorch schedulers.

    See https://pytorch.org/docs/stable/optim.html for more details.
    ---------------------------------------------------------------------------
    """

    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0,
                             epoch + epoch_count - n_epochs) / float(
                             n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=lr_decay_iters,
                                        gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5
                                                   )
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=n_epochs,
                                                   eta_min=0
                                                   )
    else:
        return NotImplementedError('learning rate policy [%s] is not '
                                   'implemented', lr_policy)
    return scheduler
# -----------------------------------------------------------------------------


def initialize_net(net,
                   init_gain,
                   gpu_ids=None,
                   gan=None):
    """
    ---------------------------------------------------------------------------
    Initialize the input net - initialization is 'normal' with specified input 
    gain.

    :param net:         input net 
    :param init_gain:   gain for normal initialization
    :param gpu_ids:     the gpus to which the net is assigned.
    :return: 
    ---------------------------------------------------------------------------
    """

    # Assign GPUs to the network tensor
    if gpu_ids is not None:
        assert(torch.cuda.is_available())

        if not isinstance(gpu_ids, list):
            gpu_ids = [gpu_ids]

        net.to(gpu_ids[0])

        if gan=='dist':
            net = torch.nn.parallel.DistributedDataParallel(net,
                                                            device_ids=gpu_ids,
                                                            output_device=gpu_ids[0],
                                                            broadcast_buffers=False)
        else:
            net = torch.nn.DataParallel(net, gpu_ids)

    # define an initialization function to be applied to the net --------------
    def initialize(block):
        """
        -----------------------------------------------------------------------
        Build an initialization function for net. The type of initialization is 
        normal and the initialization gain must be provided. 
        
        :param block:   
        :return: 
        -----------------------------------------------------------------------
        """
        
        # figure if the network is convolutional or linearly connected layer
        b_name = block.__class__.__name__
        conv_or_linear = b_name.find('Conv')!=-1 or b_name.find('Linear')!=-1 
        
        # initialization changes as per type of net
        if hasattr(block, 'weight') and conv_or_linear:
            # init.normal_(block.weight.data, 0.0, init_gain)
            
            # if hasattr(block, 'bias') and block.bias is not None:
            #     init.constant_(block.bias.data, 0.0)
            pass
        
        elif b_name.find('BatchNorm2d') !=-1:
            # BatchNorm Layer's weight is not a matrix; only normal 
            # distribution applies.
            # init.uniform_(block.weight.data)
            init.normal_(block.weight.data, 1.0, init_gain)
            init.constant_(block.bias.data, 0.0)

        # elif b_name.find('InstanceNorm2d') !=-1:
        #     # BatchNorm Layer's weight is not a matrix; only normal
        #     # distribution applies.
        #     init.normal_(block.weight.data, 1.0, init_gain)
        #     init.constant_(block.bias.data, 0.0)
    # -------------------------------------------------------------------------
    
    net.apply(initialize)
    
    return net
# -----------------------------------------------------------------------------


def calculate_gradient_penalty(netD,
                               real_data,
                               fake_data,
                               device,
                               type='mixed',
                               constant=1.0,
                               lambda_gp=1.0,
                               d_args=None):
    """
    ---------------------------------------------------------------------------
    Calculate the gradient penalty loss, used in WGAN-GP paper
    https://arxiv.org/abs/1704.00028

    :param netD (network)            - discriminator network
    :param real_data (tensor array)  - real images
    :param fake_data (tensor array)  - generated images from the generator
    :param device (str)              - GPU / CPU:
                                        from torch.device(
                                        'cuda:{}'.format(self.gpu_ids[0])
                                        )
                                        if self.gpu_ids
                                        else torch.device('cpu')
    :param type (str)                - if we mix real and fake data or not
                                       [real | fake | mixed].
    :param constant (float)          - the constant used in formula
                                       (||gradient||_2 - constant)^2
    :param lambda_gp (float)         - weight for this loss

    :return Returns the gradient penalty loss
    ---------------------------------------------------------------------------
    """

    if lambda_gp > 0.0:
        # either use real images, fake images, or a linear interpolation of two.
        if type == 'real':
            interpolatesv = real_data

        elif type == 'fake':
            interpolatesv = fake_data

        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0],
                                 real_data.nelement() //
                                 real_data.shape[0]).contiguous().view(
                                 *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)

        else:
            raise NotImplementedError('{} not implemented'.format(type))

        interpolatesv.requires_grad_(True)

        if d_args is not None:
            disc_interpolates = netD(interpolatesv, d_args)
        else:
            disc_interpolates = netD(interpolatesv)

        if isinstance(disc_interpolates, tuple):
            disc_interpolates = disc_interpolates[0]

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolatesv,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        # flat the data
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = \
            (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() \
            * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
# -----------------------------------------------------------------------------

# Classes =====================================================================


class GANLoss(nn.Module):
    """
    --------------------------------------------------------------------------
    Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label
    tensor that has the same size as the input.
    --------------------------------------------------------------------------
    """

    def __init__(self,
                 gan_mode,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 dtype=torch.float32,
                 loss_fn=None):
        """
        -----------------------------------------------------------------------
        Initialize the GANLoss class.

        Parameters:
        * gan_mode (str) - the type of GAN objective. It currently supports
                            {'vanilla', 'lsgan', 'wgangp'}
        * target_real_label (bool) - label for a real image
        * target_fake_label (bool) - label for a fake image

        Note: Do not use sigmoid as the last layer of Discriminator. LSGAN needs
        no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        -----------------------------------------------------------------------
        """
        super(GANLoss, self).__init__()

        # The label values to the register buffer
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        # Specify loss function as per type of GAN
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode =='bce':
            self.loss = nn.BCELoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        elif gan_mode =='custom':
            if loss_fn is None:
                self.loss = loss_fn
            else:
                raise IOError('Custom Loss Function not provided!')
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    # -------------------------------------------------------------------------

    def get_target_tensor(self, prediction, target_is_real):
        """
        -----------------------------------------------------------------------
        Create label tensors with the same size as the input.

        Parameters:
        :param prediction (tensor)   - typically the prediction from a
                                       discriminator
        :param target_is_real (bool) - if the ground truth label is for real
                                       images or fake images

        :return: A label tensor filled with ground truth label,
                 and with the size of the input
        -----------------------------------------------------------------------
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).type_as(prediction)
    # -------------------------------------------------------------------------

    def __call__(self, prediction, target_is_real):
        """
        -----------------------------------------------------------------------
        Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
        :param prediction (tensor)   - typically the prediction output from a
                                       discriminator
        :param target_is_real (bool) - if the ground truth label is for real
                                       images or fake images

        Returns: The calculated loss.
        -----------------------------------------------------------------------
        """

        if self.gan_mode in ['lsgan', 'vanilla', 'bce']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class NormalNLLLoss:
    """
    ---------------------------------------------------------------------------
    Calculate the negative log likelihood of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    ---------------------------------------------------------------------------
    """

    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        # print(mu.max(), var.max(), logli.max())
        nll = -(logli.sum(1).mean())

        return nll
# -----------------------------------------------------------------------------


class ImagePool():
    """
    ---------------------------------------------------------------------------
    (Adopted from pix2pix)
    This class implements an image buffer that stores previously generated
    images.

    This buffer enables us to update discriminators using a history of
    generated images rather than the ones produced by the latest generators.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pool_size):
        """
        -----------------------------------------------------------------------
        Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0,
                               no buffer will be created
        -----------------------------------------------------------------------

        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []
    # -------------------------------------------------------------------------

    def query(self, images):
        """
        -----------------------------------------------------------------------
        Return an image from the pool.

        :param images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the
                   buffer,
        and insert the current images to the buffer.
        -----------------------------------------------------------------------
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images

        return_images = []

        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # if the buffer is not full; keep inserting current images to
            # the buffer
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                # by 50% chance, the buffer will return a previously stored
                # image, and insert the current image into the buffer
                if p > 0.5:
                    # randint is inclusive
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # by another 50% chance, the buffer will return
                    # the current image
                    return_images.append(image)

        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images
    # -------------------------------------------------------------------------


class DiceLoss(nn.Module):

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = input
        target_soft = target

        # create the labels one hot tensor

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_soft, dims)
        cardinality = torch.sum(input_soft + target_soft, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


######################
# functional interface
######################


activation_layer = dict(
    relu=nn.ReLU,
    l_relu=nn.LeakyReLU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    none=None
)

normalization_layer = dict(
    batch=nn.BatchNorm2d,
    instance=nn.InstanceNorm2d,
    layer=nn.LayerNorm,
    none=None
)


# =============================================================================
# Class Ends
# =============================================================================
