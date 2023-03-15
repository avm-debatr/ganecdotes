import models.baggan.gan_util as net_util

from lib.util.visualization import *

from models.baggan.base_model import *
from models.baggan.models import StyleGANDiscriminator, StyleGANGenerator
from lib.gan.optim import conv2d_gradfix
from lib.gan.ada import AdaptiveAugment, augment

import torch.autograd as autograd
import math


class BagGANHQ(GANBaseModel):
    """
    ---------------------------------------------------------------------------
    Implements the BagGAN-HQ framework for baggage image synthesis using a
    StyleGAN2 architecture.

    The class performs the following tasks for BagGANHQ:
    - create a stylegan2 generator
    - create a stylegan2 discriminator
    - set up optimizers for training
    - set up losses for training
    - set up methods for latent vector projection
    - set up methods for latent vector manipulation

    All options are specified using the config object whose attributes provide
    the values. The configs object can be loaded as a module or can be an
    argument parser. The options in config will be described as they are used
    by the BagGAN scripts.
    ---------------------------------------------------------------------------
    """

    def __init__(self, config):
        """
        -----------------------------------------------------------------------
        Initialize BagGAN-HQ

        :param config - configuration options object.
        -----------------------------------------------------------------------
        """

        GANBaseModel.__init__(self, config)

        # ---------------------------------------------------------------------
        # specify the training losses you want to print out.
        self.loss_names = self.config.losses_to_print

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and
        # <BaseModel.load_networks>

        if self.is_train:
            self.model_names = dict(generator='G', disc='D')
        else:
            # during test time, only load G
            self.model_names = dict(generator='G')
        # ---------------------------------------------------------------------

        # define algos (both generator and discriminator) ---------------------

        self.generator = StyleGANGenerator(**self.config.generator_params)

        self.generator = net_util.initialize_net(self.generator,
                                                 self.config.init_gain,
                                                 self.gpu_ids)

        self.logger.info("Initialized Generator" + "+" * 40)
        # ---------------------------------------------------------------------

        if self.is_train:
            # define a discriminator
            self.disc = StyleGANDiscriminator(**self.config.disc_params)

            self.disc = net_util.initialize_net(self.disc,
                                                self.config.init_gain,
                                                self.gpu_ids)

            self.logger.info("Initialized Discriminator" + "+" * 40)
        # ---------------------------------------------------------------------

        # Set training parameters ---------------------------------------------
        if self.is_train:
            # define loss functions -------------------------------------------

            self.adversarial_loss = \
                net_util.GANLoss(self.config.gan_mode).to(self.device)

            if self.config.use_ppl:
                self.ppl_loss = 0.
                self.loss_d_r1 = 0.
                self.loss_g_ppl = 0.

            # -----------------------------------------------------------------
            # initialize optimizers; schedulers will be automatically
            # created by function <BaseModel.setup>.

            self.optimizer_g = torch.optim.Adam(self.generator.parameters(),
                                                lr=self.config.lr
                                                   * self.config.g_reg_ratio,
                                                betas=(self.config.beta1 ,
                                                       0.99**self.config.g_reg_ratio)
                                                )
            self.optimizer_d = torch.optim.Adam(self.disc.parameters(),
                                                lr=self.config.lr
                                                   * self.config.d_reg_ratio,
                                                betas=(self.config.beta1,
                                                       0.99 ** self.config.d_reg_ratio)
                                                )

            self.ada_aug_p = self.config.augment_p \
                            if self.config.augment_p > 0 \
                            else 0.0

            if self.config.augment and self.config.augment_p == 0:
                self.ada_augment = AdaptiveAugment(self.config.ada_target,
                                                   self.config.ada_length,
                                                   8,
                                                   self.device)

            self.optimizers.append(self.optimizer_g)
            self.optimizers.append(self.optimizer_d)
            
            self.mean_path_length = 0.

            self.mse_loss = torch.nn.MSELoss()
            self.l2_loss = net_util.GANLoss('lsgan').to(self.device)
            self.categorical_lc_loss = torch.nn.CrossEntropyLoss()
            self.continuous_lc_loss  = net_util.NormalNLLLoss()

        self.ref_image = torch.zeros(1,
                                     3,
                                     self.config.image_size,
                                     self.config.image_size).to(self.device)
        self.epoch_no = None
    # -------------------------------------------------------------------------

    def to_categorical(self, x, n_classes):
        """
        -----------------------------------------------------------------------
        Create a one-hot vector for the labels

        :param x:
        :param n_classes:
        :return:
        -----------------------------------------------------------------------
        """
        x_cat = torch.zeros(x.shape[0], n_classes)
        x_cat[range(x.shape[0]), x.long()] = 1.0

        return x_cat
    # -------------------------------------------------------------------------

    def set_input(self,
                  data_sample=None,
                  iter_no=None,
                  epoch_no=None,
                  latent=None,
                  disentangled=False,
                  gen_args=None):
        """
        -----------------------------------------------------------------------
        Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        -----------------------------------------------------------------------
        """
        im = 'ct'

        self.iter_no, self.epoch_no = iter_no, epoch_no

        if data_sample is not None:
            self.data_dims = data_sample[im].shape
            b, c, h, w     = self.data_dims
            self.bsize     = b
            self.ref_image = data_sample[im].to(self.device)
            # self.d_hist    = data_sample['hist'].to(self.device)
        else:
            b, c, h, w     = self.config.batch_size, \
                             self.config.num_channels, \
                             self.config.image_size, \
                             self.config.image_size
            self.bsize = b

        self.latent_size = self.config.w_dim if disentangled \
                                             else self.config.z_dim

        if latent is None:

            if self.config.mixing_prob > 0 and \
                    random.random() < self.config.mixing_prob:
                self.input_latent = list(torch.randn(2,
                                                 b,
                                                 self.latent_size,
                                                 device=self.device).unbind(0))
            else:
                self.input_latent = [torch.randn(b,
                                                self.latent_size,
                                                device=self.device)]
        else:
            self.input_latent = latent

        self.gen_args = gen_args
    # -------------------------------------------------------------------------

    def forward(self):
        """
        -----------------------------------------------------------------------
        Run forward pass - runs input through the generator net to produce the
        fake image.

        This function is called by:
        1) <optimize_parameters> - while train D and G nets
        2) <test> - for creating the fake image output
        -----------------------------------------------------------------------
        """
        if self.gen_args is None:
            self.out_image, self.out_latent, self.features = self.generator(self.input_latent)
        else:
            self.out_image, self.out_latent, self.features = self.generator(self.input_latent,
                                                             **self.gen_args)
    # -------------------------------------------------------------------------
    
    def calculate_perceptual_path_length_loss(self):
        """
        -----------------------------------------------------------------------
        Loss for Perceptual Path Regularization for Latent Map Smoothing.
        
        :return: 
        -----------------------------------------------------------------------
        """
        
        path_batch_size = max(1, self.config.batch_size 
                                 // self.config.path_batch_shrink)

        if self.config.mixing_prob > 0 and \
                random.random() < self.config.mixing_prob:
            l_noise = list(torch.randn(2,
                                       path_batch_size,
                                       self.latent_size,
                                   device=self.device).unbind(0))
        else:
            l_noise = [torch.randn(path_batch_size,
                                  self.latent_size,
                                  device=self.device)]

        ppl_fake_img, ppl_latents, _ = self.generator(l_noise)

        b, c, h, w = ppl_fake_img.shape
        ppl_noise = torch.randn_like(ppl_fake_img) / math.sqrt(h*w)
        # calculate gradients for noise output image w.r.t input latents
        # PPL regularizer formula from styleGAN2 paper
        grad, = autograd.grad(outputs=(ppl_fake_img * ppl_noise).sum(), 
                              inputs=ppl_latents, create_graph=True)
        
        path_lengths = torch.sqrt((grad**2).sum(2).mean(1))

        path_mean = self.mean_path_length \
                    + self.config.ppl_decay * (path_lengths.mean() 
                                               - self.mean_path_length)
        ppl_loss = ((path_lengths - path_mean)**2).mean()
        
        self.mean_path_length = path_mean.detach()
        self.path_lengths     = path_lengths

        # print(self.mean_path_length, self.path_lengths)
        
        return ppl_loss
    # -------------------------------------------------------------------------

    def calculate_r1_loss(self, ref_im):
        """
        -----------------------------------------------------------------------
        :return: 
        -----------------------------------------------------------------------
        """

        ref_im.requires_grad = True

        if self.config.augment:
            ref_im_aug, _ = augment(ref_im, self.ada_aug_p)

        else:
            ref_im_aug = ref_im

        pred_ref = self.disc(ref_im_aug)
        
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(outputs=pred_ref.sum(), 
                                       inputs=ref_im,
                                       create_graph=True)
        grad_penalty = (grad_real**2).reshape(grad_real.shape[0], 
                                              -1).sum(1).mean()

        return grad_penalty, pred_ref
    # -------------------------------------------------------------------------

    def calculate_logistic_loss(self, pred_ref, pred_fake):
        """
        -----------------------------------------------------------------------
        logistic loss
        :param real_pred:
        :param fake_pred:
        :return:
        -----------------------------------------------------------------------
        """
        real_loss = F.softplus(-pred_ref)
        fake_loss = F.softplus(pred_fake)

        return real_loss.mean() + fake_loss.mean()
        # ---------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def calculate_nonsaturating_loss(self, pred_fake):
        """
        -----------------------------------------------------------------------
        Non-saturating loss

        :param fake_pred:
        :return:
        -----------------------------------------------------------------------
        """

        loss = F.softplus(-pred_fake).mean()

        return loss
    # -------------------------------------------------------------------------
    
    def backward_d(self):
        """
        -----------------------------------------------------------------------
        Calculate GAN loss for the discriminator.
        The loss has two terms:
        1) the disc. output for (input + out sino) , i.e., the fake sino
        2) the disc. output for (input + ref sino) , i.e., the real sino

        1 is D(x) (must be 1) and 2 is D(G(z)) (must be 0)
        -----------------------------------------------------------------------
        """
        d_in_1 = self.out_image.detach()
        d_in_2 = self.ref_image

        if self.config.augment:
            d_in_1, _ = augment(d_in_1, self.ada_aug_p)
            d_in_2, _ = augment(d_in_2, self.ada_aug_p)

        # else:
        #     real_img_aug = real_img

        # Discriminator loss for fake sino (out_sino) -------------------------

        pred_fake = self.disc(d_in_1)
        # pred_fake       = self.disc(torch.cat((d_in_1, self.hist_image), 1))
        self.loss_d_out = self.adversarial_loss(pred_fake, False)
        # self.loss_d_out = self.loss_d_out + self.l2_loss(pred_fake, False)
        # self.loss_d_out = 0.5*self.loss_d_out
        # ---------------------------------------------------------------------

        # Discriminator loss for real sino (ref_sino) -------------------------

        pred_ref = self.disc(d_in_2)
        # pred_ref   = self.disc(torch.cat((d_in_2, self.hist_image), 1))
        self.loss_d_ref = self.adversarial_loss(pred_ref, True)
        # self.loss_d_ref = self.loss_d_ref + self.l2_loss(pred_ref, True)
        # self.loss_d_ref = 0.5 * self.loss_d_ref
        # self.loss_d = self.calculate_logistic_loss(pred_ref, pred_fake)

        # ---------------------------------------------------------------------

        if self.config.gan_mode=='wgangp':
            self.loss_d_gp, _ = calculate_gradient_penalty(
                self.disc,
                # torch.cat((d_in_1, self.hist_image), 1),
                # torch.cat((d_in_2, self.hist_image), 1),
                d_in_1, d_in_2,
                self.device
                )

            if type(self.loss_d_gp) != float:
                self.loss_d = (self.loss_d_out + self.loss_d_ref)*0.25 + \
                               self.loss_d_gp * 0.5
            else:
                self.loss_d = self.loss_d_out + self.loss_d_ref
        else:
            self.loss_d = self.loss_d_out + self.loss_d_ref

        self.loss_d = self.loss_d_out + self.loss_d_ref

        # combine loss and calculate gradients --------------------------------

        # self.loss_d = self.loss_d_out + self.loss_d_ref
        self.loss_d.backward()
    # -------------------------------------------------------------------------

    def backward_g(self):
        """
        -----------------------------------------------------------------------
        Calculate GAN + L1 loss for the generator

        -----------------------------------------------------------------------
        """

        # The loss term D(G(x, z)) for training generator network -------------

        d_in = self.out_image

        if self.config.augment:
            d_in, _ = augment(d_in, self.ada_aug_p)


        # pred_fake       = self.disc(torch.cat((d_in, self.hist_image), 1))
        # pred_fake       = self.disc(d_in)
        pred_fake = self.disc(d_in)

        # print(pred_fake.mean())
        self.loss_g_gan = self.adversarial_loss(pred_fake, True)
        # self.loss_g_gan = self.calculate_nonsaturating_loss(pred_fake)

        # self.loss_g_mse  = self.mse_loss(self.out_image, self.ref_image)
        self.loss_g_l2  = 0. # self.l2_loss(pred_fake, 1.)

        # Info Max Losses
        if self.iter_no %self.config.g_reg_every==0:
            self.loss_g_ppl = self.calculate_perceptual_path_length_loss()

        self.loss_g = self.loss_g_gan + self.loss_g_ppl

        self.loss_g.backward()
    # -------------------------------------------------------------------------

    def optimize_parameters(self):
        """
        -----------------------------------------------------------------------
        Runs the forward and backprop steps for one iteration of GAN training.
        This involves updating both the generator and discriminator - the two
        algos have different losses. During update, the discriminator is
        first updated followed by generator.

        :return:
        -----------------------------------------------------------------------
        """
        self.forward()

        # update D ------------------------------------------------------------

        # enable backprop for D

        self.set_requires_grad(self.disc, True)
        self.optimizer_d.zero_grad()    # set D's gradients to zero

        self.backward_d()               # calculate gradients for D
        self.optimizer_d.step()         # update D's weights

        if self.config.augment and self.config.augment_p == 0:
            self.ada_aug_p = self.ada_augment.tune(self.loss_d_ref)
            self.r_t_stat  = self.ada_augment.r_t_stat

        if self.iter_no % self.config.d_reg_every:

            self.loss_d_r1, pred_ref = \
                self.calculate_r1_loss(self.ref_image)

            self.disc.zero_grad()

            self.loss_d_r1 = self.config.r1_lambda / 2 \
                             * self.loss_d_r1 \
                             * self.config.d_reg_every \
                             + 0 * pred_ref[0]

            self.loss_d_r1.backward()
            self.optimizer_d.step()         # update D's weights

        # update G ------------------------------------------------------------

        # D requires no gradients when optimizing G

        self.set_requires_grad(self.disc, False)

        self.optimizer_g.zero_grad()  # set G's gradients to zero
        self.backward_g()             # calculate gradients for G

        self.optimizer_g.step()       # update G's weights
    # -------------------------------------------------------------------------

    def test(self):
        """
        -----------------------------------------------------------------------
        Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save
        intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization
        results
        -----------------------------------------------------------------------
        """

        with torch.no_grad():
            self.forward()
            return self.out_image.detach()
    # -------------------------------------------------------------------------
