from lib.util.util import *

import math
import torch
import torch.nn as nn
import copy

from models.baggan.blocks import PixelNorm, EqualizedLinearBlock
from models.baggan.blocks import ConstantInputBlock, ConvLayerWithStyleMod, \
                                 SynthesisBlock,     XToImageConverter
from models.baggan.blocks import DiscConvBlock, ResidualBlock


DEFAULT_CHL_MULTIPLIER = 2

# DEFAULT_RES_TO_CHANNEL_MAP =  {4:   512,
#                                8:   512,
#                                16:  512,
#                                32:  256 * DEFAULT_CHL_MULTIPLIER,
#                                64:  128 * DEFAULT_CHL_MULTIPLIER,
#                                128: 64  * DEFAULT_CHL_MULTIPLIER,
#                                256: 32  * DEFAULT_CHL_MULTIPLIER,
#                                512: 16  * DEFAULT_CHL_MULTIPLIER}

DEFAULT_RES_TO_CHANNEL_MAP =  {4:   512,
                               8:   512,
                               16:  256 * DEFAULT_CHL_MULTIPLIER,
                               32:  128 * DEFAULT_CHL_MULTIPLIER,
                               64:  64  * DEFAULT_CHL_MULTIPLIER,
                               128: 32  * DEFAULT_CHL_MULTIPLIER,
                               256: 16  * DEFAULT_CHL_MULTIPLIER,
                               512: 8   * DEFAULT_CHL_MULTIPLIER}


class LatentMappingNetwork(nn.Module):
    """
    ---------------------------------------------------------------------------
    8-layer MLP that converts the uniformly sampled vector z to the
    disentangled vector w.
    ---------------------------------------------------------------------------
    """
    def __init__(self,
                 latent_dims,
                 n_layers,
                 lr_multiplier):
        """
        -----------------------------------------------------------------------
        :param latent_dims:     input-output dimensions for MLP
        :param n_layers:        number of layers
        :param lr_multiplier:   learning rate multiplier
        -----------------------------------------------------------------------
        """

        super().__init__()
        z_dim, w_dim = latent_dims

        layers = [EqualizedLinearBlock(z_dim,
                                       w_dim,
                                       lr_multiplier,
                                       activation='lrelu')]

        layers = layers + [EqualizedLinearBlock(w_dim,
                                                w_dim,
                                                lr_multiplier,
                                                activation='lrelu')
                  for _ in range(1, n_layers)]

        layers = [PixelNorm()] + layers

        self.mapper = nn.Sequential(*layers)
    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------
        :param x:
        :return:
        -----------------------------------------------------------------------
        """

        return self.mapper(x)
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class StyleGANGenerator(nn.Module):

    def __init__(self,
                 latent_dims,
                 img_resolution,
                 mlp_layers=8,
                 mlp_lr=0.01,
                 img_chls=1,
                 fir_filter=[1,3,3,1],
                 res2chlmap=None):
        """
        -----------------------------------------------------------------------
        :param latent_dims:
        :param img_resolution:
        :param mlp_layers:
        :param mlp_lr:
        :param img_chls:
        :param fir_filter:
        :param res2chlmap:
        -----------------------------------------------------------------------
        """

        assert img_resolution > 16 and img_resolution & (img_resolution-1) == 0
        super().__init__()

        self.z_dim, self.w_dim = latent_dims

        # ZtoW Latent Mapper --------------------------------------------------
        self.style = LatentMappingNetwork(latent_dims,
                                          mlp_layers,
                                          mlp_lr)

        # Synthesis Network ---------------------------------------------------
        init_res = 4

        if res2chlmap is None:
            self.res2chlmap = DEFAULT_RES_TO_CHANNEL_MAP

        # first block : constant input block
        self.const_input_block = ConstantInputBlock(self.res2chlmap[init_res])

        # Starting blocks for synthesizer -------------------------------------

        # second block: initial convolution block
        self.conv_init = SynthesisBlock(self.res2chlmap[init_res],
                                        self.res2chlmap[init_res],
                                        3, # kernel width
                                        self.w_dim,
                                        mode='none',
                                        fir_filter=fir_filter)

        # x to image converter: for initial block
        self.x_to_img_init  = XToImageConverter(self.res2chlmap[init_res],
                                                3,
                                                self.w_dim,
                                                upsample=False,
                                                fir_filter=fir_filter)

        # log of the image reosluton require dfor calculations
        self.res_log = int(math.log(img_resolution, 2))
        
        # number of layers required = 2 x [log(im_res)-2] +1
        # - subtracted by 2 because because min_res in const block is 4=2**2
        # - twice the number of layers for mod-demod
        # - 1 more layer for initial convolution
        
        self.num_layers = (self.res_log - 2) * 2 + 1

        self.conv_blks     = nn.ModuleList()
        self.upsample_blks = nn.ModuleList()
        self.x_to_img_blks = nn.ModuleList()
        self.noise_blks    = nn.Module()

        in_chls = self.res2chlmap[4]

        # Create noise blocks for all layers - noise at end of the each block
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]

            self.noise_blks.register_buffer(f"noise_{layer_idx}",
                                            torch.randn(*shape))

        # Start from the third block onwards with the skip connections,
        # the mod-demod conv blocks and noise blks
        for i in range(3, self.res_log + 1):
            out_chls = self.res2chlmap[2 ** i]

            # create a synthesis block with upsampling at given scale
            self.conv_blks.append(SynthesisBlock(in_chls,
                                                 out_chls,
                                                 3,
                                                 self.w_dim,
                                                 mode='up',
                                                 fir_filter=fir_filter))

            # create a synthesis block w/o upsampling at given scale 
            # (post upsampling)
            self.conv_blks.append(SynthesisBlock(out_chls,
                                                 out_chls,
                                                 3,
                                                 self.w_dim,
                                                 fir_filter=fir_filter))
            
            # X-to-Img Conversion Blocks for skip connections
            self.x_to_img_blks.append(XToImageConverter(out_chls,
                                                        3,
                                                        self.w_dim))

            in_chls = out_chls

        # self.head = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
        #                           # nn.Conv2d(1, 1, 3, 1, 1),
        #                           nn.Tanh())
        # self.head_pe = nn.Sequential(nn.Conv2d(out_chl, 1, 3, 1, 1),
        #                              # nn.Conv2d(1, 1, 3, 1, 1),
        #                              nn.Tanh())
        self.head_m = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
                                    nn.Conv2d(1, 1, 3, 1, 1),
                                    nn.Conv2d(1, 1, 3, 1, 1),
                                    nn.Conv2d(1, 1, 3, 1, 1),
                                    nn.Tanh())

        self.n_latent = self.res_log * 2 - 2
    # -------------------------------------------------------------------------

    # def make_noise(self):
    #     device = self.input.input.device
    # 
    #     noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
    # 
    #     for i in range(3, self.res_log + 1):
    #         for _ in range(2):
    #             noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
    # 
    #     return noises

    # def mean_latent(self, x, n_latent=500):
    #     """
    #     ------------------------------------------------------------------------
    #     :param n_latent:
    #     :return:
    #     ------------------------------------------------------------------------
    #     """
    #     latent_in = torch.randn(n_latent,
    #                             self.w_dim,
    #                             device=x.device)
    #     return self.style(latent_in).mean(0, keepdim=True)

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.w_dim, device=self.const_input_block.const_block.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, x):
        return self.style(x)

    def make_noise(self):
        device = self.const_input_block.const_block.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.res_log + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def forward(self,
                styles,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                noise=None,
                randomize_noise=True):
        """
        -----------------------------------------------------------------------

        :param styles:
        :param return_latents:
        :param inject_index:
        :param truncation:
        :param truncation_latent:
        :param input_is_latent:
        :param noise:
        :param randomize_noise:
        :return:
        -----------------------------------------------------------------------
        """

        # Check if the input in z or w latent space, 
        # if in z latent space, convert to w latent space using style mapper
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        # If no noise provided, generate noise for each scale
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noise_blks, f"noise_{i}") for i in range(self.num_layers)
                ]
        
        # if no latent for truncation is provided, use the mean value 
        # for w space
        if truncation_latent is None:
            truncation_latent = self.calculate_mean_latent(styles[0])

        # Perform truncation trick if truncation < 1,
        # make sure truncation_latent is provided for that
        if truncation < 1:
            style_t = []

            for s in styles:
                style_t.append(truncation_latent 
                               + truncation * (s - truncation_latent))
            styles = style_t

        # Repeat noise vector for the number of layers
        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            # Style Mixing
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent  = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1,
                                                    self.n_latent - inject_index,
                                                    1)

            latent = torch.cat([latent, latent2], 1)

        out = self.const_input_block(latent)
        # print('1:', out.detach().min().item(), out.detach().max().item())

        out = self.conv_init(out,
                             latent[:, 0],
                             noise=noise[0])
        # print('2:', out.detach().min().item(), out.detach().max().item())

        skip = self.x_to_img_init(out, latent[:, 1])
        # print('3:', skip.detach().min().item(), skip.detach().max().item())

        features = [out]

        i = 1
        for conv1, conv2, n1, n2, x_to_img_init in zip(self.conv_blks[::2],
                                                  self.conv_blks[1::2],
                                                  noise[1::2],
                                                  noise[2::2],
                                                  self.x_to_img_blks
        ):
            # conv block with upsampling - takes in prev output, upsamples and fwds
            out = conv1(out, latent[:, i], noise=n1)
            features.append(out)
            # conv block w/o upsampling - produces x output for current block
            out = conv2(out, latent[:, i + 1], noise=n2)
            # print(i, ':', out.detach().min().item(), out.detach().max().item())
            features.append(out)

            # skip connection block for msg - combines x output and skip connection 
            # from prev block
            skip = x_to_img_init(out, latent[:, i + 2], skip)
            # print(i, ':', skip.detach().min().item(), skip.detach().max().item())

            i += 2

        image = skip

        # out_image = self.head(image)
        # mask      = self.head_m(image)
        #
        # image = torch.cat((image[:,0:1,:,:], mask), 1)
        # image = image[:,0:1,:,:]

        if return_latents:
            return image, latent

        else:
            return image, features
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


DEFAULT_RES_TO_CHANNEL_MAP =  {4:   512,
                               8:   256 * DEFAULT_CHL_MULTIPLIER,
                               16:  128 * DEFAULT_CHL_MULTIPLIER,
                               32:  64  * DEFAULT_CHL_MULTIPLIER,
                               64:  32  * DEFAULT_CHL_MULTIPLIER,
                               128: 16  * DEFAULT_CHL_MULTIPLIER,
                               256: 8   * DEFAULT_CHL_MULTIPLIER,
                               512: 4   * DEFAULT_CHL_MULTIPLIER}


class StyleGANDiscriminator(nn.Module):
    def __init__(self, 
                 img_resolution,
                 img_chls,
                 out_chls=1,
                 res2chlmap=None,
                 with_q=False,
                 q_args=None):
        """
        -----------------------------------------------------------------------


        :param img_resolution:
        :param img_chls:
        :param fir_filter:
        :param res2chlmap:
        :param with_q:
        :param q_args:      {'q_layers', 'n_cat_c', 'n_cont_c'}
        -----------------------------------------------------------------------
        """
        super().__init__()

        if res2chlmap is None:
            self.res2chlmap = DEFAULT_RES_TO_CHANNEL_MAP

        self.img_res  = img_resolution
        self.img_chls = img_chls
        self.with_q   = with_q

        if self.with_q:
            self.q_args = q_args.copy()

        convs = [DiscConvBlock(img_chls,
                               self.res2chlmap[self.img_res], 1)]

        res_log = int(math.log(self.img_res, 2))
        in_chls = self.res2chlmap[self.img_res]

        for i in range(res_log, 2, -1):
            out_chls = self.res2chlmap[2 ** (i - 1)]
            convs.append(ResidualBlock(in_chls, out_chls))
            in_chls = out_chls

        self.std_group, self.std_feat = 4, 1

        if not self.with_q:
            self.convs = nn.Sequential(*convs)
            self.final_conv = DiscConvBlock(in_chls + 1, self.res2chlmap[4], 3)
            self.final_linear = nn.Sequential(
                EqualizedLinearBlock(self.res2chlmap[4] * 4 * 4,
                                     self.res2chlmap[4],
                                     activation="lrelu"),
                EqualizedLinearBlock(self.res2chlmap[4], 
                                     1,
                                     activation='linear'),
            )

        else:
            
            # break the convolutional blocks into adversarial layer and disc layer
            # self.convs_adv = nn.Sequential(*convs[:-self.q_args['q_layers']-1])
            self.convs_adv = nn.Sequential(*convs[:-self.q_args['q_layers']])

            # Also append a Q layer for Info Max
            self.convs_d = nn.Sequential(*convs[self.q_args['q_layers']:])

            # Specify final layers for dis and q models

            self.final_conv_d = DiscConvBlock(in_chls + 1, self.res2chlmap[4], 3)
            self.final_linear_d = nn.Sequential(
                EqualizedLinearBlock(self.res2chlmap[4] * 4 * 4,
                                     self.res2chlmap[4],
                                     activation="lrelu"),
                EqualizedLinearBlock(self.res2chlmap[4], 1, activation='linear'),
            )

            # Specify final layers for q network (categorical)
            if self.q_args['n_cat_c']>0:
                self.convs_q_cat        = copy.deepcopy(self.convs_d)
                self.final_conv_q_cat   = copy.deepcopy(self.final_conv_d)
                self.final_linear_q_cat = nn.Sequential(
                                EqualizedLinearBlock(self.res2chlmap[4] * 4 * 4,
                                                     self.res2chlmap[4],
                                                     activation="lrelu"),
                                EqualizedLinearBlock(self.res2chlmap[4], 
                                                     self.q_args['n_cat_c']
                                                     *self.q_args['n_classes'],
                                                     activation='linear'),
                                nn.Softmax()
                            )

            # Specify final layers for q network (continuous)
            if self.q_args['n_cont_c'] > 0:
                self.convs_q_cont = copy.deepcopy(self.convs_d)
                self.final_conv_q_cont = copy.deepcopy(self.final_conv_d)
                self.final_linear_q_cont = nn.Sequential(
                                EqualizedLinearBlock(self.res2chlmap[4] * 4 * 4,
                                                     self.res2chlmap[4],
                                                     activation="lrelu"),
                                EqualizedLinearBlock(self.res2chlmap[4],
                                                     self.q_args['n_cont_c']*2,
                                                     activation='linear'),
                                nn.Tanh()
                            )

    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------

        :param x:
        :return:
        -----------------------------------------------------------------------
        """
        
        if not self.with_q:
            # Pass input through the residual blocks
            x = self.convs(x)
    
            b, c, h, w = x.shape
            # group = min(b, self.std_group)
            group = b
            # calculate std of a bathc of channels from output
            std = x.view(group, -1, self.std_feat, c // self.std_feat, h, w)
            std = torch.sqrt(std.var(0, unbiased=False) + 1e-8)
            std = std.mean([2, 3, 4], keepdims=True).squeeze(2)
            std = std.repeat(group, 1, h, w)
    
            # append the calculated std to the output
            out = torch.cat([x, std], 1)
            out = self.final_conv(out)
            out = out.view(b, -1)
    
            out = self.final_linear(out)
    
            return out
        else:
            
            x = self.convs_adv(x)
            
            # output for discriminator
            x_d = self.convs_d(x)
            b, c, h, w = x_d.shape
            std_d = self.get_std_from_output(x_d)

            # append the calculated std to the output
            out_d = torch.cat([x_d, std_d], 1)
            out_d = self.final_conv_d(out_d)
            out_d = out_d.view(b, -1)
            out_d = self.final_linear_d(out_d)

            # output for categorical q network
            out_q_cat = None

            if self.q_args['n_cat_c']>0:
                x_q_cat = self.convs_q_cat(x)
                b, c, h, w = x_q_cat.shape
                std_q_cat = self.get_std_from_output(x_q_cat)
    
                # append the calculated std to the output
                out_q_cat = torch.cat([x_q_cat, std_q_cat], 1)
                out_q_cat = self.final_conv_q_cat(out_q_cat)
                out_q_cat = out_q_cat.view(b, -1)
                out_q_cat = self.final_linear_q_cat(out_q_cat)

            # output for continuous q network
            out_q_cont = None

            if self.q_args['n_cont_c'] > 0:
                x_q_cont = self.convs_q_cont(x)
                b, c, h, w = x_q_cont.shape
                std_q_cont = self.get_std_from_output(x_q_cont)

                # append the calculated std to the output
                out_q_cont = torch.cat([x_q_cont, std_q_cont], 1)
                out_q_cont = self.final_conv_q_cont(out_q_cont)
                out_q_cont = out_q_cont.view(b, -1)
                out_q_cont = self.final_linear_q_cont(out_q_cont)

            return out_d, out_q_cat, out_q_cont
    # -------------------------------------------------------------------------
    
    def get_std_from_output(self, x):
        """
        -----------------------------------------------------------------------
        Calculate standard deviation for a batch of channels

        :param x:
        :return:
        -----------------------------------------------------------------------
        """
        b, c, h, w = x.shape
        group = min(b, self.std_group)
        # calculate std of a batch of channels from output
        std = x.view(group, -1, self.std_feat, c // self.std_feat, h, w)
        std = torch.sqrt(std.var(0, unbiased=False) + 1e-8)
        std = std.mean([2, 3, 4], keepdims=True).squeeze(2)
        std = std.repeat(group, 1, h, w)

        return std
# -----------------------------------------------------------------------------

