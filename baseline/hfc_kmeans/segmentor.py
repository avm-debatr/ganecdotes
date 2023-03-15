from torch import nn
from baseline.hfc_kmeans.hfc_kmeans_clustering import \
                                       FlatKMeansHFC, \
                                       HierarchicalKMeansHFC,\
                                       multi_sample_hierarchical_encoding,\
                                       hierarchical_label_encoding

from lib.oneshot.image_augmentor import *


class HFCPreprocessor(object):

    def __init__(self,
                 model,
                 model_config,
                 perturb_args,
                 hfc_args,
                 hfc_algo='hfc_kmeans',
                 hier_encode=True,
                 hle_samples=500,
                 train=True,
                 out_dir=None,
                 logger=None
                 ):
        """
        -----------------------------------------------------------------------
        Preprocesses the data before feeding to the one-shot segmentor.


        :param model:           Pretrained StyleGAN model
        :param perturb_args:    arguments+config for perturbation
        :param hfc_args:        init args for HFC model
        :param hfc_algo:        which HFC clustering algo to use:
                                {'hfc_kmeans' | 'hfc_kmeans_hier'}
        :param hier_encode:     whether to add hierarchical encoding
        :param train:           train or load from saved
        :param out_dir:         output directory
        -----------------------------------------------------------------------
        """

        self.model          = model
        self.perturb_config = perturb_args
        self.hfc_args       = hfc_args
        self.hier_encode    = hier_encode
        self.hfc_algo       = hfc_algo
        self.out_dir        = out_dir
        self.train          = train
        self.logger         = logger
        self.model_config   = model_config
        self.hle_samples    = hle_samples

        assert self.hfc_algo in ['hfc_kmeans', 'hfc_kmeans_hier']

        self.hfc_args['base_args']['out_dir'] = self.out_dir
        self.hfc_args['base_args']['logger'] = self.logger

        if self.hfc_algo=='hfc_kmeans':
            self.hfc_model = FlatKMeansHFC(**hfc_args)

        if self.hfc_algo == 'hfc_kmeans_hier':
            self.hfc_model = HierarchicalKMeansHFC(**hfc_args)

        if self.hier_encode:
            self.belief_file = os.path.join(self.out_dir, 'beliefs.npz')
        self.trained_beliefs = None
    # -------------------------------------------------------------------------
        
    def train_hfc_model(self,
                        input_latent,
                        return_aug=False):
        """
        -----------------------------------------------------------------------
        - Augment the image generated from input latent
        - Use hidden features from augmented samples to train the HFC model
        - save model + outputs

        :param input_latent:
        :return: trained model (sklearn.cluster.KMeans)
        -----------------------------------------------------------------------
        """

        # input latent, z to extended latents, w
        with torch.no_grad():
            mean_latent = self.model.mean_latent(
                                    self.model_config.num_latents_for_mean)
            truncation  = self.perturb_config['truncation']
            input_latent = input_latent.unsqueeze(0) \
                               if len(input_latent.shape)==2  \
                               and input_latent.shape[0]>1\
                               else input_latent

            orig_img, w_latents = self.model([input_latent],
                                              return_latents=True,
                                              truncation_latent=mean_latent,
                                              truncation=truncation,
                                              input_is_latent=True)
            test_w_latents = w_latents.detach().clone()

        hidden_features = []
        perturb_init = [0] * (2 * self.perturb_config['n_layers'])

        new_latent_list = []
        # ---------------------------------------------------------------------

        # create augmented images by perturbing w latent vectors
        with torch.no_grad():

            for k in range(self.perturb_config['n_layers']):
                # perturb the latent for current layer
                perturb_stds = perturb_init.copy()
                perturb_stds[2 * k] = perturb_stds[2 * k + 1] \
                                    = self.perturb_config['perturb_std'][k]

                perturbed_latents = \
                    create_perturbed_vectors_from_latents(
                                    test_w_latents,
                                    self.model,
                                    n_samples=self.perturb_config['n_samples'],
                                    n_layers=self.perturb_config['n_layers'],
                                    perturb_std=perturb_stds
                    )

                # change latent for layer
                new_latents = test_w_latents.repeat(
                                    self.perturb_config['n_samples'], 1, 1)
                new_latents[:, 2 * k, :] = perturbed_latents[2 * k]
                new_latents[:, 2 * k + 1, :] = perturbed_latents[2 * k + 1]

                new_latent_list.append(perturbed_latents[2 * k])
                new_latent_list.append(perturbed_latents[2 * k + 1])

                # hidden features for nth layers
                perturbed_img, hfeat = \
                    create_images_and_features_from_perturbed_latents(
                        new_latents,
                        self.model,
                        {'truncation': truncation,
                         'mean_latent': mean_latent},
                        layer_no=k,
                        return_feat=True,
                        return_image=True,
                        skip_const=True
                    )

                hidden_features.append(hfeat)
                torch.cuda.empty_cache()

                self.logger.info(f"Generated features for Layer: {k}")
        # ---------------------------------------------------------------------

        if self.hfc_algo=='hfc_kmeans_hier':
            self.hfc_model.hierarchical_fit(hidden_features)
        else:
            self.hfc_model.fit(hidden_features)

        if self.hier_encode:
            self.trained_beliefs = multi_sample_hierarchical_encoding(
                self.model,
                self.hfc_model,
                self.hle_samples,
                self.perturb_config['n_layers'])

            np.savez_compressed(self.belief_file, self.trained_beliefs)

        if return_aug:
            return hidden_features, new_latent_list
    # -------------------------------------------------------------------------

    def predict_hfc_vectors(self, input_latent):
        """
        -----------------------------------------------------------------------
        use trained K-Means models to predict hfc vectors from GAN hidden
        features

        :param input_latent: latent vector for image
        :return: hfc vector for image (B x sum(clusters_per_layer) x H x W )
        -----------------------------------------------------------------------
        """

        # predict feature vectors from trained HFC model
        with torch.no_grad():
            mean_latent = self.model.mean_latent(
                self.model_config.num_latents_for_mean)
            truncation  = self.perturb_config['truncation']

            orig_img, w_latents = self.model([input_latent],
                                        return_latents=True,
                                        truncation_latent=mean_latent,
                                        truncation=truncation,
                                        input_is_latent=True)
            test_w_latents = w_latents.detach().clone()

            perturbed_img, hfeat = \
                create_images_and_features_from_perturbed_latents(
                    test_w_latents.repeat(1, 1, 1),
                    self.model,
                    {'truncation': 0.7,
                     'mean_latent': mean_latent},
                    return_feat=True,
                    return_image=True,
                    skip_const=True
                )

            out_preds, out_labels =  self.hfc_model.predict(
                            hfeat[:self.perturb_config['n_layers']])

        if not self.train and self.hier_encode:
            self.trained_beliefs = np.load(self.belief_file,
                                           allow_pickle=True)['arr_0']

        if self.hier_encode:

            hier_labels, hier_preds, _ = \
                hierarchical_label_encoding(out_labels,
                    out_preds[:,
                    -self.hfc_model.clusters_per_layer[-1]:, :, :],
                    self.hfc_model.clusters_per_layer,
                    self.trained_beliefs)
            hier_preds = torch.cat(hier_preds[::-1], 1)

            # hier_preds = hier_preds.repeat(1, 2, 1, 1)
            # hier_preds[:,::2,:,:] = out_preds

            hier_preds = hier_preds*2-1

            return hier_preds, hier_labels

        out_preds = out_preds*2-1

        return out_preds, out_labels
    # -------------------------------------------------------------------------


class OneShotSegmentor(nn.Module):

    def __init__(self,
                 in_ch,
                 n_class,
                 size='S'):
        """
        -----------------------------------------------------------------------
        One Shot Segmentor processing hfc feature vectors for mask prediction

        :param in_ch:       len(clusters_per_layer)
        :param n_class:     classes for prediction
        :param size:        segmentor architecture
                            Check:
                            Rewatbowornwong, Pitchaporn, Nontawat Tritrong,
                            and Supasorn Suwajanakorn. "Repurposing GANs for
                            One-shot Semantic Part Segmentation."
                            IEEE Transactions on Pattern Analysis and Machine
                            Intelligence (2022).

        -----------------------------------------------------------------------
        """
        super().__init__()

        assert size in ['S', 'M', 'L']

        dilations = {
            'S': [1, 2, 1, 2, 1],
            'M': [1, 2, 4, 1, 2, 4, 1],
            'L': [1, 2, 4, 8, 1, 2, 4, 8, 1],
        }[size]

        channels = {
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
    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------
        :param x: input hfc image (1 x nclusters x H x W)
        :return:
        -----------------------------------------------------------------------
        """
        return self.layers(x)
    # -------------------------------------------------------------------------
