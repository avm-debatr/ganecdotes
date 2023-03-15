import torch.nn.functional as F
import torch.nn as nn
import math

from lib.oneshot.image_augmentor import *
from lib.util.visualization import quick_imshow

try:    from apex.parallel.LARC import LARC
except: from apex.parallel.LARC import LARC

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimCLRClustering(object):

    def __init__(self,
                 model,
                 model_config,
                 perturb_args,
                 simclr_args,
                 logger=None,
                 train=True,
                 out_dir=None,
                 device='cuda',
                 tb=None,
                 layer_hf_dim=None):
        """
        -----------------------------------------------------------------------
        Cluster HFC features with SimCLR Contrastive loss
        Loss formulation adopted from:
        Chen, Ting, et al. "A simple framework for contrastive learning of
        visual representations." International conference on machine learning.
        PMLR, 2020.

        :param model:           GAN model
        :param model_config:    GAN params
        :param perturb_args:    for latent vector perturbation
        :param simclr_args:     for simclr loss
        :param logger:          to log prompts
        :param out_dir:         borrow from src.OneShotPipeline
        :param tb:              Tensorboard summary writer
        :param layer_hf_dim:    Hidden Feat. Dimiensions for each GAN layer
        -----------------------------------------------------------------------
        """

        self.model = model
        self.model_config = model_config
        self.perturb_args = perturb_args
        self.simclr_args = simclr_args
        self.device = device
        self.writer = tb

        self.nclasses    = simclr_args['nclasses']

        self.logger = logger
        self.train = train
        self.out_dir = out_dir

        self.swav_dir = os.path.join(self.out_dir, 'simclr')
        os.makedirs(self.swav_dir, exist_ok=True)
        
        self.projection_file = os.path.join(self.out_dir, 'projection.pt')

        if not self.train:
            
            if os.path.exists(self.projection_file):
                self.projection = torch.load(self.projection_file)
            else:
                self.logger.info("Projection File not found - pretraining ...")
        
        self.softmax_loss = nn.Softmax()

        with torch.no_grad():
            self.mean_latent = self.model.mean_latent(
                                    self.model_config.num_latents_for_mean)
            self.truncation  = self.model_config.truncation

        self.fixed_transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        self.layer_hf_dim = layer_hf_dim
        self.similarity = nn.CosineSimilarity(dim=0)
            
    # -------------------------------------------------------------------------

    def create_pixel_feature_vectors(self, features, pred=False):
        """
        -----------------------------------------------------------------------
        Scale and concatenate all hidden features into a single vector
        :param features: 
        :return: 
        -----------------------------------------------------------------------
        """

        if pred:
            features = [f.to('cpu') for f in features]
        h = max([f.shape[-2] for f in features])
        w = max([f.shape[-1] for f in features])

        features = torch.cat([F.interpolate(f, (h, w),
                                            mode=self.simclr_args['hf_interp'])
                          for f in features], dim=1)[:,
                                                     :self.simclr_args['hlen'],
                                                     :,
                                                     :]

        return features.to(self.device)
    # -------------------------------------------------------------------------

    def preprocess(self, input_latent):
        """
        -----------------------------------------------------------------------
        overloaded method for all precprocessors

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
                 num_test_samples=2):
        """
        -----------------------------------------------------------------------
        This is where the self-supervised learning happens

        :return: 
        -----------------------------------------------------------------------        
        """
        num_iters  = self.simclr_args['num_iters']
        batch_size = num_patches = self.simclr_args['batch_size']
        temperature = self.simclr_args['temperature']

        # SimCLR with Projection Layer ========================================

        self.projection = nn.Sequential(
                                    nn.Linear(self.simclr_args['hlen'],
                                              self.nclasses,
                                              bias=False),
                                    nn.BatchNorm1d(self.nclasses),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(self.nclasses,
                                              self.nclasses,
                                              bias=False),
                                    # nn.BatchNorm1d(self.nclasses),
                                    # nn.Tanh()
                                    ).to(self.device)
                
        self.logger.info("Projection Network:")
        self.logger.info(self.projection.__str__())
        
        self.logger.info("SimCLR Parameters:")
        for k,v in self.simclr_args.items():
            self.logger.info(f"{k}: {v}")

        optimizer = torch.optim.SGD(self.projection.parameters(),
                                    **self.simclr_args['train_args'])
        
        optimizer = LARC(optimizer=optimizer,
                         trust_coefficient=self.simclr_args['trust_coeff'],
                         clip=False)
        # =====================================================================

        t0 = time.time()

        for e in range(num_iters):

            train_latent = torch.randn(1,
                                       self.model_config.latent_dim
                                       ).to(self.device)
            train_latent = self.model.style(train_latent)

            # first transformation for train samples
            hidden_feat_s, _ = \
                self.create_hidden_features_from_perturbed_vectors(
                    input_latent=train_latent,
                    layer_no=self.perturb_args['layer_no']
                )
            hidden_feat_s = self.fixed_transforms(hidden_feat_s)
            hidden_feat_s = F.normalize(hidden_feat_s, dim=1)

            # pixelwise feat
            flat_hfeat_s = torch.squeeze(hidden_feat_s).flatten(1)

            # second tf
            hidden_feat_t, _ = \
                self.create_hidden_features_from_perturbed_vectors(
                    input_latent=train_latent,
                    layer_no=self.perturb_args['layer_no']
                )
            hidden_feat_t = self.fixed_transforms(hidden_feat_t)
            hidden_feat_t = F.normalize(hidden_feat_t, dim=1)

            flat_hfeat_t = torch.squeeze(hidden_feat_t).flatten(1)

            # picks a batch pf pixels from both tf
            b, c, h, w = hidden_feat_s.shape
            picks      = torch.randperm(h * w)

            flat_hfeat_s = flat_hfeat_s[:, picks]
            flat_hfeat_s = flat_hfeat_s[:, :self.simclr_args['batch_size']]

            flat_hfeat_t = flat_hfeat_t[:, picks]
            flat_hfeat_t = flat_hfeat_t[:, :self.simclr_args['batch_size']]
            
            flat_hfeat = torch.zeros(flat_hfeat_s.shape[0],
                                     2*self.simclr_args['batch_size']).to(self.device)

            # arrange all pixelwise samples
            flat_hfeat[:, ::2]  = flat_hfeat_s
            flat_hfeat[:, 1::2] = flat_hfeat_t

            loss_matrix = torch.zeros(2*self.simclr_args['batch_size'],
                                      2*self.simclr_args['batch_size'])
            loss_matrix = loss_matrix.to(self.device)

            # similarity matrix for loss calculation
            similarity_matrix = torch.zeros(2*self.simclr_args['batch_size'],
                                            2*self.simclr_args['batch_size'])
            similarity_matrix = similarity_matrix.to(self.device)

            # scores for view pair
            scores = self.projection(flat_hfeat.t()).t()

            for i in range(2*self.simclr_args['batch_size']):
                for j in range(2*self.simclr_args['batch_size']):
                    
                    scores_s, scores_t = scores[i], scores[j]

                    similarity_matrix[i, j] = self.similarity(scores_s,
                                                              scores_t)

            similarity_matrix /= temperature
            # similarity_matrix += 1e-09

            # loss for view pairs
            for i in range(2*self.simclr_args['batch_size']):
                for j in range(2*self.simclr_args['batch_size']):

                    numl = torch.exp(similarity_matrix[i,j])

                    denl = torch.exp(similarity_matrix[i,:])
                    denl = torch.stack([d
                            for ino, d in enumerate(denl)
                            if ino!=i]).sum()

                    loss_matrix[i, j] = -torch.log(numl/denl)

            # total loss
            loss = torch.stack([loss_matrix[2*k-1, 2*k]
                                + loss_matrix[2*k, 2*k-1]
                        for k in range(self.simclr_args['batch_size'])]).sum()
            loss *= 1/(2*batch_size)

            self.writer.add_scalar('simclr/loss', loss, e)

            # Step 4: SGD update - updates the prototype matrix
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if e%self.simclr_args['iter_print_freq']==0:
            self.logger.info(f" (Iter:{e}):"
                             f"\tLoss: {loss:.03f},"
                             f"\tTime: {time.time()-t0:.03f}"
                             f"\t")
            torch.cuda.empty_cache()

        torch.save(self.projection, self.projection_file)
    # -------------------------------------------------------------------------
    
    def create_hidden_features_from_perturbed_vectors(self,
                                                      layer_no=None,
                                                      input_latent=None,
                                                      input_is_latent=True):
        """
        -----------------------------------------------------------------------
        Image augmentation via latent vector perturbation

        :return: 
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

            _, w_latents = self.model([input_latent],
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
            # self.logger.info(f"Generated features for Layer: {layer_no}")

            hfeat = self.create_pixel_feature_vectors(hfeat)
            # hfeat[:,:sum(self.layer_hf_dim[layer_no]),:,:] = 0

        return hfeat, perturbed_img
    # -------------------------------------------------------------------------

    def predict_simclr_codes(self, input_latent):
        """
        -----------------------------------------------------------------------
        gives simclr codes during inference

        :param input_latent:
        :return:
        -----------------------------------------------------------------------
        """

        with torch.no_grad():

            input_latent = input_latent.unsqueeze(0)\
                           if len(input_latent.shape)==2 \
                                and input_latent.shape[0]>1 \
                           else input_latent

            perturbed_img, hfeat = self.model([input_latent],
                                   truncation=self.model_config.truncation,
                                   truncation_latent=self.mean_latent,
                                   input_is_latent=True,
                                   randomize_noise=False)

            torch.cuda.empty_cache()

        hfeat = self.create_pixel_feature_vectors(hfeat)
        hfeat = F.normalize(hfeat, dim=1)
        
        flat_hfeat = torch.squeeze(hfeat).flatten(1)
        scores = self.projection(flat_hfeat.t()).t()

        b, c, h, w = hfeat.shape        
        new_shape = (b, self.nclasses, h, w)
        
        scores = scores.reshape(new_shape)

        out_preds = scores
        out_labels = out_preds.max(1)[1]

        return out_preds, out_labels
    # -------------------------------------------------------------------------


class OneShotSegmentor(nn.Module):
    """
    Network for fine-tuning during one-shot learning
    """

    def __init__(self,
                 in_ch,
                 n_class,
                 size='S'):
        super().__init__()

        assert size in ['XS','S', 'M', 'L', 'Lin']

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
