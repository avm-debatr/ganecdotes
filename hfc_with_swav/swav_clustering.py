from lib.util.util import *

from PIL import Image
from sklearn.cluster import * # KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from skimage.measure import regionprops
from lib.oneshot.image_augmentor import *
from torch.nn import init
from torchvision.utils import make_grid

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math

from lib.oneshot.image_augmentor import *
from lib.util.visualization import quick_imshow
from scipy.stats import norm

try:
    from apex.parallel.LARC import LARC
except:
    from apex.parallel.LARC import LARC

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SwAVClustering(object):

    def __init__(self,
                 model,
                 model_config,
                 perturb_args,
                 swav_args,
                 sinkhorn_args,
                 logger=None,
                 train=True,
                 out_dir=None,
                 device='cuda',
                 tb=None,
                 layer_hf_dim=None):
        """
        -----------------------------------------------------------------------
        :param model:               StyleGAN model
        :param model_config:        model config class
        :param perturb_args:        parmeters for GAN layer perturbation
        :param swav_args:           parameters for SwaAV loss calculation
        :param sinkhorn_args:       parameters for Sinkhorn-Knopp iterations
        :param logger:              borrows from src.OneShotPipeline
        :param train:               if training or using pressaved models
        :param out_dir:             borrows from src.OneShotPipeline
        :param device:              torch.device for ss models
        :param tb:                  Tensorboard summary writer
        :param layer_hf_dim:        hidden feature length for every GAN layer
        -----------------------------------------------------------------------
        """

        self.model = model
        self.model_config = model_config
        self.perturb_args = perturb_args
        self.swav_args = swav_args
        self.device = device
        self.writer = tb

        self.nclasses    = swav_args['nclasses']
        self.nprototypes = swav_args['nprototypes']
        self.niters   = sinkhorn_args['niters']
        self.eps      = sinkhorn_args['eps']

        self.sinkhorn_args = sinkhorn_args.copy()

        self.logger = logger
        self.train = train
        self.out_dir = out_dir

        self.swav_dir = os.path.join(self.out_dir, 'swav')
        os.makedirs(self.swav_dir, exist_ok=True)
        
        self.prototype_file = os.path.join(self.out_dir, 'prototypes.pt')
        self.projection_file = os.path.join(self.out_dir, 'projection.pt')

        if not self.train:
            
            if os.path.exists(self.projection_file):
                self.projection = torch.load(self.projection_file)
                self.prototype = torch.load(self.prototype_file)

            else:
                self.logger.info("Prototype File not found - pretraining ...")
        
        self.softmax_loss = nn.Softmax()

        with torch.no_grad():
            self.mean_latent = self.model.mean_latent(
                                    self.model_config.num_latents_for_mean)
            self.truncation  = self.model_config.truncation

        self.fixed_transforms = transforms.Compose([
            transforms.RandomRotation(10),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        self.layer_hf_dim = layer_hf_dim
            
    # -------------------------------------------------------------------------

    def create_pixel_feature_vectors(self, features, pred=False):
        """
        -----------------------------------------------------------------------
        Upscale and concatenate hidden featuee to create a create single
        pixelwise vector
        :param features: 
        :return: 
        -----------------------------------------------------------------------
        """

        if pred:
            features = [f.to('cpu') for f in features]
        h = max([f.shape[-2] for f in features])
        w = max([f.shape[-1] for f in features])

        features = torch.cat([F.interpolate(f, (h, w),
                                            mode=self.swav_args['hf_interp'])
                          for f in features], dim=1)[:,
                                                     :self.swav_args['hlen'],
                                                     :,
                                                     :]

        return features.to(self.device)
    # -------------------------------------------------------------------------

    def get_swav_codes_from_hidden_features(self,
                                            hfeat,
                                            new_shape=None,
                                            picks=None,
                                            train=False):
        """
        -----------------------------------------------------------------------
        Pass hidden feature vectors through porjection + prototype layers

        :param hfeat:       hidden feature vectors
        :param new_shape:   if output needs to be reshaped to image dimensions
        :param picks:       if random pixels/patches  need to be picked
        :param train:       training or inference
        :return:
        -----------------------------------------------------------------------
        """

        if self.swav_args['sampling_method'] == 'patch':

            if picks is not None:
                hfeat = hfeat[:,
                              :,
                              picks:picks+self.swav_args['patch_size'],
                              picks:picks+self.swav_args['patch_size']]

            flat_hfeat = torch.squeeze(hfeat).flatten(1)

        elif self.swav_args['sampling_method'] == 'random':

            flat_hfeat = torch.squeeze(hfeat).flatten(1)

            if picks is not None:

                flat_hfeat = flat_hfeat[:, picks]
                flat_hfeat = flat_hfeat[:, :self.swav_args['patch_size']]

        # scores = self.prototype(flat_hfeat.t()).t()

        scores = self.projection(flat_hfeat.t())

        if train:
            scores = F.normalize(scores, p=2, dim=1)
            scores = self.prototype(scores)
        else:
            scores = scores.t()

        if new_shape is not None:
            scores = scores.reshape(new_shape)

        return scores

    # -------------------------------------------------------------------------

    def preprocess(self, input_latent):
        """
        -----------------------------------------------------------------------
        Overloaded function for preprocessors

        :return:
        -----------------------------------------------------------------------
        """

        if self.train:
            self.pretrain(input_latent)
        else:

            if os.path.exists(self.projection_file) and not self.train:
                pass
            else:
                self.pretrain(input_latent)
    # -------------------------------------------------------------------------

    def pretrain(self,
                 input_latent,
                 num_test_samples=5):
        """
        -----------------------------------------------------------------------
        This is where self-supervised learning with SwAV happens

        :param input_latent - input latent - placeholder for other
                              preprocessors
        :param num_test_samples - no of test sample to plot at every epoch

        :return: 
        -----------------------------------------------------------------------        
        """
        num_epochs  = self.swav_args['num_epochs']
        num_samples = self.swav_args['num_samples']
        num_patches = self.swav_args['num_patches']
        temperature = self.swav_args['temperature']

        test_latent = [torch.randn(1,
                            self.model_config.latent_dim).to(self.device)
                       for _ in range(num_test_samples)]

        with torch.no_grad():
            test_imgs = []

            for i in range(num_test_samples):

                test_img, _ = self.model([test_latent[i]],
                                          return_latents=True,
                                          truncation_latent=self.mean_latent,
                                          truncation=self.truncation,
                                          input_is_latent=False)
                t = 0.5*(test_img.clamp(-1, 1) + 1)
                test_img = transforms.ToPILImage()(t[0,:,:,:]/test_img.max())
                test_imgs.append(test_img)

        # SwAV with Projection + Prototype Layers =============================

        if self.swav_args['projn_nw']=='linear':
            self.projection = nn.Sequential(
                                        nn.Linear(self.swav_args['hlen'],
                                                  self.nclasses,
                                                  bias=False),
                                        ).to(self.device)
        elif self.swav_args['projn_nw']=='1-layer':
            self.projection = nn.Sequential(
                                        nn.Linear(self.swav_args['hlen'],
                                                  self.nclasses,
                                                  bias=False),
                                        nn.LeakyReLU(inplace=True),
                                        ).to(self.device)
        elif self.swav_args['projn_nw'] == '2-layer':
            self.projection = nn.Sequential(
                                        nn.Linear(self.swav_args['hlen'],
                                                  self.nclasses,
                                                  bias=False),
                                        nn.BatchNorm1d(self.nclasses),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(self.nclasses,
                                                  self.nclasses,
                                                  bias=False),
                                        nn.BatchNorm1d(self.nclasses),
                                        nn.Tanh()
                                        ).to(self.device)
        self.prototype = nn.Linear(self.nclasses,
                                   self.nprototypes).to(self.device)
        
        self.logger.info("Projection Network:")
        self.logger.info(self.projection.__str__())
        self.logger.info("Prototype Matrix:")
        self.logger.info(self.prototype.__str__())
        
        self.logger.info("SwAV Parameters:")
        for k,v in self.swav_args.items():
            self.logger.info(f"{k}: {v}")

        self.logger.info("Sinkhorn-Knopp Algo. Parameters:")
        for k,v in self.sinkhorn_args.items():
            self.logger.info(f"{k}: {v}")

        optimizer = torch.optim.SGD(list(self.projection.parameters()) + \
                                     list(self.prototype.parameters()),
                                     **self.swav_args['train_args'])
        
        optimizer = LARC(optimizer=optimizer,
                         trust_coefficient=self.swav_args['trust_coeff'],
                         clip=False)
        # =====================================================================

        # SwAV with Prototype Layers Only =====================================
        # self.prototype = nn.Linear(self.swav_args['hlen'],
        #                            self.nclasses).to(self.device)

        # optimizer = torch.optim.Adam(self.prototype.parameters(),
        #                              **self.swav_args['train_args'])
        # =====================================================================

        warmup_lr_schedule = np.linspace(self.swav_args['start_warmup'],
                                         self.swav_args['base_lr'],
                                         num_samples *
                                         self.swav_args['warmup_epochs'])

        iters = np.arange(num_samples * (self.swav_args['num_epochs']
                                         - self.swav_args['warmup_epochs']))
        cosine_lr_schedule = np.array([self.swav_args['final_lr']
                                       + 0.5 * (self.swav_args['base_lr']
                                                - self.swav_args['final_lr'])
               * (1 + math.cos(math.pi * t / ((self.swav_args['num_epochs']
                               - self.swav_args['warmup_epochs']))))
                   for t in iters])
        lr_schedule = np.concatenate((warmup_lr_schedule,
                                      cosine_lr_schedule))
        t0 = time.time()

        for e in range(num_epochs):

            for i in range(num_samples):
                train_latent = torch.randn(1,
                                           self.model_config.latent_dim
                                           ).to(self.device)
                train_latent = self.model.style(train_latent)

                with torch.no_grad():
                    w = self.prototype.weight.data.clone()
                    w = F.normalize(w, dim=1, p=2)
                    self.prototype.weight.copy_(w)

                if self.swav_args['use_scheduler']:
                    iteration = e*num_samples + i
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_schedule[iteration]

                # Step 1: Generate augmented versions of input image samples
                #         Augmentation is done by perturbing latent vector
                #         of input image
                #         Perturbing different layers generates different
                #         augmentations features generated for augmented
                #         samples created from StyleGAN
                # Dim - (1 x C x H x W) , C is the sum of channel lengths
                # of all hidden features

                hidden_feat_s, perturbed_img_s, l_s = \
                    self.create_hidden_features_from_perturbed_vectors(
                        input_latent=train_latent,
                        layer_no=self.perturb_args['layer_no']
                    )
                hidden_feat_t, perturbed_img_t, l_t = \
                    self.create_hidden_features_from_perturbed_vectors(
                        input_latent=train_latent,
                        layer_no=self.perturb_args['layer_no']
                    )

                hidden_feat_s = self.fixed_transforms(hidden_feat_s)
                hidden_feat_t = self.fixed_transforms(hidden_feat_t)

                img_s = torch.norm(hidden_feat_s, p=2, dim=1)
                img_t = torch.norm(hidden_feat_t, p=2, dim=1)

                # Here SwAV is clustering the pixel feature vectors of
                # the image for segmentation. Hence, each pixel is a data
                # sample for clustering thus creating a batch of H x W
                # samples.
                # Hidden features are therefore flattening to create a
                # C x D vector set where D = H*W (or patch size)

                # Step 2: Scores for the hidden features based on the
                #         prototype

                b, c, h, w = hidden_feat_s.shape
                loss = 0.

                for pno in range(num_patches):

                    if self.swav_args['patch_size'] is None \
                            or self.swav_args['patch_size']==h:
                        pick_t = pick_s = None
                    else:
                        if self.swav_args['sampling_method']=='patch':
                            pick_s = np.random.choice(
                                       h-self.swav_args['patch_size'])

                        elif self.swav_args['sampling_method']=='random':
                            pick_s = torch.randperm(h * w)

                        pick_t = pick_s

                    scores_s = self.get_swav_codes_from_hidden_features(
                                hidden_feat_s,
                                picks=pick_s,
                                train=True
                               )
                    scores_t = self.get_swav_codes_from_hidden_features(
                                hidden_feat_t,
                                picks=pick_t,
                                train=True
                               )

                    with torch.no_grad():
                        q_ns = self.sinkhorn_knopp(scores_s, img_s)
                        q_nt = self.sinkhorn_knopp(scores_t, img_t)

                    # Softmax probabilities for scores
                    p_s = scores_s/temperature
                    p_t = scores_t/temperature

                    # Step 3: calculating swapped prediction loss

                    global_loss = self.calculate_swapped_prediction_loss(p_s,
                                                                         p_t,
                                                                         q_ns,
                                                                         q_nt)

                    loss = loss + global_loss

                    if self.swav_args['add_local_loss']:
                        m_hfeat_s = hidden_feat_s
                        m_hfeat_t = hidden_feat_t

                        m_hfeat_s[:,:sum(self.layer_hf_dim[l_s]),:,:] = 0
                        m_hfeat_t[:,:sum(self.layer_hf_dim[l_t]),:,:] = 0

                        mscores_s = self.get_swav_codes_from_hidden_features(
                            m_hfeat_s,
                            picks=pick_s,
                            train=True
                        )
                        mscores_t = self.get_swav_codes_from_hidden_features(
                            m_hfeat_t,
                            picks=pick_t,
                            train=True
                        )

                        with torch.no_grad():
                            mq_ns = self.sinkhorn_knopp(mscores_s, img_s)
                            mq_nt = self.sinkhorn_knopp(mscores_t, img_t)

                        # Softmax probabilities for scores
                        mp_s = mscores_s / temperature
                        mp_t = mscores_t / temperature

                        local_loss = self.calculate_swapped_prediction_loss(mp_s,
                                                                            mp_t,
                                                                            mq_ns,
                                                                            mq_nt)

                        loss = loss + local_loss

                loss = loss/self.swav_args['num_patches']

                self.writer.add_scalar('swav/loss', loss, e)

                # Step 4: SGD update - updates the prototype matrix
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if e%self.swav_args['epoch_print_freq']==0:
                self.logger.info(f" E:{e}\t|"
                                 f"\tLoss: {loss:.03f} \t|"
                                 f"\tT: {time.time()-t0:.03f}")

            if self.swav_args['plot_test_images']:

                np_masks = min(self.nclasses, self.swav_args['max_masks'])

                out_l = []

                out_ms = [[] for _ in range(np_masks)]

                for i in range(num_test_samples):
                    out_pred, out_labels = \
                        self.predict_swav_codes(test_latent[i], False)

                    out = transforms.ToPILImage()(
                          out_labels[0, :, :] / out_labels.max())
                    out_l.append(out)

                    for i in range(np_masks):
                        out_ms[i].append(
                    out_pred[0, i, :, :].detach().cpu().numpy()
                        )

                self.writer.add_image('swav/test_image', grid)

                ims = test_imgs+out_l

                for o in out_ms:
                    ims = ims + o

                quick_imshow(np_masks+2, num_test_samples,
                             ims,
                             colorbar=False,
                             colormap='gray')

                plt.savefig(os.path.join(self.swav_dir, f'test_epoch_{e}.png'))
                plt.close()

        self.logger.info(f"Finished pretraining - Saving projection file")
        torch.save(self.prototype, self.prototype_file)
        torch.save(self.projection, self.projection_file)

    # -------------------------------------------------------------------------
    
    def sinkhorn_knopp(self, scores, img):
        """
        -----------------------------------------------------------------------
        Sinkhorn-Knopp iterations for douvle normalization

        :param scores:  input scores from image features (C^T * Z)
        :param img:     img if using image histogram pdf
        :return:  return the calculated cluster assignments
        -----------------------------------------------------------------------
        """
        Q = torch.exp(scores/self.eps).T
        Q /= torch.sum(Q)
        K, B = Q.shape

        if self.sinkhorn_args['source_pdf']=='image':
            histb = torch.histc(img, B) + 1e-9
            histb[0] = histb[1]
            histb = histb/histb.sum()

            histk = torch.histc(img, K) + 1e-9
            histk[0] = histk[1]
            histk = histk/histk.sum()

            u, r, c = torch.zeros(K), histk, histb

        else:
            u, r, c = torch.zeros(K), torch.ones(K)/K, torch.ones(B)/B

        u, r, c = u.to(self.device), r.to(self.device), c.to(self.device)
        
        for _ in range(self.niters):
            u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).T
    # -------------------------------------------------------------------------
        
    def calculate_swapped_prediction_loss(self, 
                                          softmax_ns, 
                                          softmax_nt,
                                          code_ns, 
                                          code_nt):
        """
        -----------------------------------------------------------------------
        calculate SwAV loss for two image view

        :param softmax_ns:  predicted code for s view
        :param softmax_nt:  predicted code for t view
        :param code_ns:     calculated (SK) code for s view
        :param code_nt:     calculated (SK) code for t view
        :return: 
        -----------------------------------------------------------------------
        """
        
        lst = torch.mean(torch.sum(code_ns*F.log_softmax(softmax_nt,
                                                         dim=1),
                                   dim=1))
        lts = torch.mean(torch.sum(code_nt*F.log_softmax(softmax_ns,
                                                         dim=1),
                                   dim=1))
        return -0.5*(lst + lts)

    # -------------------------------------------------------------------------
    
    def create_hidden_features_from_perturbed_vectors(self,
                                                      layer_no=None,
                                                      input_latent=None,
                                                      input_is_latent=True):
        """
        -----------------------------------------------------------------------
        for image augmentation via layer perturbation

        :param layer_no         - layer to perturb, if none it is
                                  randomly selected
        :param input_latent     - input latent vector
        :param input_is_latent  - whether input is z or w latent

        :return: hfeat, perturbed_img, layer_no -
                 returns hidden feature vectors, the perturbed image and the
                 perturbed layer number
        -----------------------------------------------------------------------
        """

        with torch.no_grad():
            if input_latent is None:
                input_latent = torch.randn(1,
                            self.model_config.latent_dim).to(self.device)
                input_is_latent = False

            input_latent = input_latent.unsqueeze(0) \
                           if len(input_latent.shape)==2 \
                              and input_latent.shape[0]>1 \
                                 else input_latent
            orig_img, w_latents = self.model([input_latent],
                                             return_latents=True,
                                             truncation_latent=self.mean_latent,
                                             truncation=self.truncation,
                                             input_is_latent=input_is_latent)
            test_w_latents = w_latents.detach().clone()

        if layer_no is None:
            layer_no = np.random.choice(
                                list(range(self.perturb_args['n_layers'])))

        perturb_init = [0] * (2 * self.perturb_args['n_layers'])

        # ---------------------------------------------------------------------

        # create augmented images by perturbing w latent vectors
        with torch.no_grad():
            perturb_stds = perturb_init.copy()
            perturb_stds[2 * layer_no] \
                = perturb_stds[2 * layer_no + 1] \
                = self.perturb_args['perturb_std'][layer_no]

            perturbed_latents = \
                create_perturbed_vectors_from_latents(
                    test_w_latents,
                    self.model,
                    n_samples=self.perturb_args['n_samples'],
                    n_layers=self.perturb_args['n_layers'],
                    perturb_std=perturb_stds
                )

            # change latent for layer
            new_latents = test_w_latents.repeat(
                self.perturb_args['n_samples'], 1, 1)
            new_latents[:,
                2 * layer_no, :]     = perturbed_latents[2 * layer_no]
            new_latents[:,
                2 * layer_no + 1, :] = perturbed_latents[2 * layer_no + 1]

            # hidden features for nth layers
            perturbed_img, hfeat = \
                create_images_and_features_from_perturbed_latents(
                    new_latents,
                    self.model,
                    {'truncation':  self.truncation,
                     'mean_latent': self.mean_latent},
                    return_feat=True,
                    return_image=True
                )

            torch.cuda.empty_cache()
            hfeat = self.create_pixel_feature_vectors(hfeat)

        return hfeat, perturbed_img, layer_no
    # -------------------------------------------------------------------------

    def predict_swav_codes(self, input_latent, input_is_latent=True):
        """
        -----------------------------------------------------------------------
        obtain swav codes for input latent vector at inference time

        :param input_latent: latent vector x
        :param input_is_latent: whether input is z or w latent vector

        :return: out_preds, out_labels -both hard/soft swav cluster predictions
        -----------------------------------------------------------------------
        """

        with torch.no_grad():

            perturbed_img, hfeat = self.model([input_latent],
                                       truncation=self.model_config.truncation,
                                       truncation_latent=self.mean_latent,
                                       input_is_latent=True,
                                       randomize_noise=False)

            torch.cuda.empty_cache()

        hfeat = self.create_pixel_feature_vectors(hfeat)

        b, c, h, w = hfeat.shape

        out_preds = self.get_swav_codes_from_hidden_features(hfeat,
                                                             (b,
                                                              self.nclasses,
                                                              h,
                                                              w),
                                                             train=False)
        out_labels = out_preds.max(1)[1]

        return out_preds, out_labels
    # -------------------------------------------------------------------------


class OneShotSegmentor(nn.Module):
    """
    Fine-tuning network for one-shot segmentation
    """

    def __init__(self,
                 in_ch,
                 n_class,
                 size='S'):
        super().__init__()

        assert size in ['XXS','XS','S', 'M', 'L', 'Lin']

        if size=='Lin':

            self.layers = nn.Sequential(nn.Linear(in_ch, n_class),
                                        nn.LeakyReLU(0.2, inplace=True)
                                        )
        else:
            dilations = {
                'XXS': [1],
                'XS': [1, 2, 1],
                'S': [1, 2, 1, 2, 1],
                'M': [1, 2, 4, 1, 2, 4, 1],
                'L': [1, 2, 4, 8, 1, 2, 4, 8, 1],
            }[size]

            channels = {
                'XXS': [12],
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
