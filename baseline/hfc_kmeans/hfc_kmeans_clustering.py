from lib.util.util import *

from PIL import Image
from sklearn.cluster import *
from skimage.measure import regionprops
from lib.oneshot.image_augmentor import *

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaseHFCModel(object):

    def __init__(self,
                 out_dir,
                 n_layers=6,
                 clusters_per_layer=[],
                 out_size=128,
                 presaved=False,
                 logger=None
                 ):
        """
        -----------------------------------------------------------------------
        Base class for hidden feature clustering

        :param out_dir:             output directory where models are saved
        :param n_layers:            number of layers processed
        :param clusters_per_layer:  cluster labels assigned for each layer
        :param out_size:            output size image dimensions
        :param presaved:            load presaved models
        -----------------------------------------------------------------------
        """

        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.n_layer = n_layers
        self.clusters_per_layer = clusters_per_layer
        self.out_size = out_size
        self.presaved = presaved
        self.logger   = logger

        self.model_fpaths = [os.path.join(self.out_dir,
                                          f"clusterer_layer_{n}.sav")
                             for n in range(self.n_layer)]

        self.stats_file = os.path.join(self.out_dir, 'model_stats.npz')

        if self.presaved and os.path.exists(self.stats_file):
            stats = np.load(self.stats_file, allow_pickle=True)

            self.means = stats['means']
            self.stds  = stats['stds']
        else:
            self.stds = self.means = [None] * len(self.clusters_per_layer)

        if self.presaved:
            if all([os.path.exists(fp)for fp in self.model_fpaths]):
                self.clusterers = [pickle.load(open(fpath, 'rb'))
                                   for fpath in self.model_fpaths]
            else:
                raise FileNotFoundError('Models not found - '
                                        'use BaseHFCModel.fit() '
                                        'to create model first!')
    # -------------------------------------------------------------------------

    def fit(self, hidden_feat):
        """
        -----------------------------------------------------------------------
        Fit clustering model for each layer given the layerwise hidden features

        :param hidden_feat:
        :return:
        -----------------------------------------------------------------------
        """
        assert len(hidden_feat)==self.n_layer

        self.clusterers = []

        for n in range(self.n_layer):
            model = self._layerwise_fit(hidden_feat[n], n)
            self.clusterers.append(model)
            self.save_model(model, n)

            if self.logger is None:
                print(f"Fitted model for Layer {n}")
            else:
                self.logger.info(f"Fitted model for Layer {n}")

        np.savez_compressed(self.stats_file,
                            means=self.means,
                            stds=self.stds)

    # -------------------------------------------------------------------------

    def predict(self, hidden_feat):
        assert len(hidden_feat) == self.n_layer

        cluster_maps = []
        cluster_labels = []

        for n in range(self.n_layer):

            c_labels, c_maps = self._layerwise_predict(hidden_feat[n], n)

            cluster_maps.append(c_maps)
            cluster_labels.append(c_labels)

        cluster_maps = torch.cat(cluster_maps, 1)

        return cluster_maps, cluster_labels
    # -------------------------------------------------------------------------

    def _layerwise_fit(self, feat, n):
        return
    # -------------------------------------------------------------------------

    def _layerwise_predict(self, feat, n):
        return None, None
    # -------------------------------------------------------------------------

    def save_model(self, model, n):

        pickle.dump(model, open(self.model_fpaths[n], 'wb'))
    # -------------------------------------------------------------------------


class FlatKMeansHFC(BaseHFCModel):

    def __init__(self,
                 kmeans_args,
                 base_args):
        """
        -----------------------------------------------------------------------
        Use K-Means to perform hidden feature clustering

        :param kmeans_args: sklearn kmenas args
        :param base_args:   constructor args fro BaseHFCModel
        -----------------------------------------------------------------------
        """

        self.kmeans_args = kmeans_args.copy()
        BaseHFCModel.__init__(self, **base_args)

    # -------------------------------------------------------------------------

    def _layerwise_fit(self, feat, n):
        """
        -----------------------------------------------------------------------
        :param feat:    hidden feature - dim must match layer no
        :param n:       layer no
        :return:        sklearn.cluster.KMeans
        -----------------------------------------------------------------------
        """
        clusterer = KMeans(n_clusters=self.clusters_per_layer[n],
                           **self.kmeans_args)

        hfeat = feat.permute(1, 0, 2, 3).flatten(1).cpu().numpy().T

        self.means[n], self.stds[n] = hfeat.mean(axis=0), \
                                      hfeat.std(axis=0)

        # hfeat = (hfeat - hfeat.mean(axis=0))/hfeat.std(axis=0)

        clusterer.fit(hfeat)

        return clusterer
    # -------------------------------------------------------------------------

    def _layerwise_predict(self, feat, n):
        """
        -----------------------------------------------------------------------
        :param feat:    hidden feature - dim must match layer no
        :param n:       layer no
        :return:        cluster labels for image pixels
        -----------------------------------------------------------------------
        """
        b, c, h, w = feat.shape
        hfeat = feat.permute(1, 0, 2, 3).flatten(1).cpu().numpy().T

        # hfeat = (hfeat - self.means[n])/self.stds[n]

        clusterer = self.clusterers[n]

        labels = clusterer.predict(hfeat)
        scores = clusterer.transform(hfeat)

        labels = torch.from_numpy(labels)
        # scores = torch.from_numpy(scores.T)

        # labels = labels.reshape(h, w, b).permute(2, 0, 1)
        labels = labels.reshape(1, b, h, w).permute(1, 0, 2, 3)
        # scores = scores.reshape(self.clusters_per_layer[n],
        #                         b, h, w).permute(1, 0, 2, 3)

        label_maps = torch.zeros(b, self.clusters_per_layer[n], h, w)

        for k in range(self.clusters_per_layer[n]):
            label_maps[:, k:k+1, :, :][labels==k] = 1
        # label_maps = scores
        # label_maps = (label_maps - label_maps.min()) \
        #              / (label_maps.max() - label_maps.min())

        label_maps = transforms.Resize(size=(self.out_size,
                                             self.out_size),
                                       interpolation=Image.NEAREST)\
                                      (label_maps)

        return labels, label_maps
    # -------------------------------------------------------------------------


class HierarchicalKMeansHFC(BaseHFCModel):

    def __init__(self,
                 kmeans_args,
                 base_args):
        """
        -----------------------------------------------------------------------
        Use K-Means to perform hierarchical hidden feature clustering

        :param kmeans_args: sklearn kmenas args
        :param base_args:   constructor args fro BaseHFCModel
        -----------------------------------------------------------------------
        """

        self.kmeans_args = kmeans_args.copy()
        BaseHFCModel.__init__(self, **base_args)
    # -------------------------------------------------------------------------

    def hierarchical_fit(self, hidden_feat):
        """
        -----------------------------------------------------------------------
        Fit clustering model hierarchically using K-Means for each layer
        given the layerwise hidden features

        :param hidden_feat:
        :return:
        -----------------------------------------------------------------------
        """
        assert len(hidden_feat) == self.n_layer

        self.clusterers = []
        self._cluster_centers = None

        for n in range(self.n_layer):

            model = self._layerwise_fit(hidden_feat[n], n)

            if n != self.n_layer-1:
                self._cluster_centers = self.calculate_cluster_centers(
                    hidden_feat[n],
                    hidden_feat[n+1],
                    model.labels_,
                    n+1
                )

            self.clusterers.append(model)
            self.save_model(model, n)

            if self.logger is None:
                print(f"Fitted model for Layer {n}")
            else:
                self.logger.info(f"Fitted model for Layer {n}")

    # -------------------------------------------------------------------------

    def _layerwise_fit(self, feat, n):
        """
        -----------------------------------------------------------------------
        :param feat: input feature - dims must match layer no
        :param n: layer no
        :return: sklearn.cluster.KMeans model
        -----------------------------------------------------------------------
        """

        best_clf = None
        best_inertia = float("inf")

        for i in range(1):

            if self._cluster_centers is None:
                clusterer = KMeans(n_clusters=self.clusters_per_layer[n],
                                   **self.kmeans_args)
            else:

                if i!=0:
                    ctrs = self._cluster_centers \
                           + np.random.randn(*self._cluster_centers.shape)

                else:
                    ctrs = self._cluster_centers

                clusterer = KMeans(n_clusters=self.clusters_per_layer[n],
                                   init=ctrs,
                                   **self.kmeans_args)

            hfeat = feat.permute(1, 0, 2, 3).flatten(1).cpu().numpy().T
            clusterer.fit(hfeat)

            if clusterer.inertia_ < best_inertia:
                best_clf = clusterer
                best_inertia = clusterer.inertia_

            # self.logger.info(f"Initialization No : {i}")

        return best_clf
    # -------------------------------------------------------------------------

    def _layerwise_predict(self, feat, n):
        """
        -----------------------------------------------------------------------
        predict cluster for a given layer

        :param feat:    hidden features - dim must match layer no
        :param n:       layyer no
        :return: cluster predictions
        -----------------------------------------------------------------------
        """
        b, c, h, w = feat.shape
        hfeat = feat.permute(1, 0, 2, 3).flatten(1).cpu().numpy().T

        clusterer = self.clusterers[n]

        labels = clusterer.predict(hfeat)
        labels = torch.from_numpy(labels)
        # labels = labels.reshape(h, w, b).permute(2, 0, 1)
        labels = labels.reshape(1, b, h, w).permute(1, 0, 2, 3)

        label_maps = torch.zeros(b, self.clusters_per_layer[n], h , w)

        for k in range(self.clusters_per_layer[n]):
            label_maps[:, k:k+1, :, :][labels==k] = 1

        label_maps = transforms.Resize(size=(self.out_size,
                                             self.out_size),
                                       interpolation=Image.NEAREST)\
                                      (label_maps)

        return labels, label_maps
    # -------------------------------------------------------------------------

    def calculate_cluster_centers(self,
                                  feat_old,
                                  feat_new,
                                  labels,
                                  n):
        """
        -----------------------------------------------------------------------
        propagate cluster centers from previous layer to current layer with new
        number of clusters

        used in hierarchical clustering.

        :param feat_old:    hidden features from older layer
        :param feat_new:    hidden features for new layer
        :param labels:      labels assigned to pixel in older layer
        :param n:           layer no for which cluster center are propagated
        :return:            labels x len(feat_new) - cluster centers

        Used as initial value for KMeans in the next layer
        -----------------------------------------------------------------------
        """

        b, c, h, w = feat_old.shape

        labels = torch.from_numpy(labels).to(feat_old.device)
        labels = labels.reshape(1, b, h, w).permute(1, 0, 2, 3)

        bn, cn, hn, wn = feat_new.shape
        labels = transforms.Resize((hn, wn),
                                   interpolation=Image.NEAREST)(labels)

        ffeat   = feat_new.permute(1, 0, 2, 3).flatten(1)
        flabels = labels.permute(1, 0, 2, 3).flatten(1)

        label_vals = torch.unique(labels)

        cluster_centers = torch.zeros(self.clusters_per_layer[n-1],
                                      cn)

        for lbl in label_vals:
            lbl_vect = ffeat[:, flabels[0]==lbl]
            cluster_centers[int(lbl),:] = lbl_vect.mean()

        cluster_centers = cluster_centers.cpu().numpy()
        cluster_centers = np.repeat(cluster_centers,
                                    2,
                                    axis=0)

        return cluster_centers
    # -------------------------------------------------------------------------


def hierarchical_label_encoding(im_labels,
                                one_hot_label,
                                clusters_per_layer, beliefs=None):
    """
    ---------------------------------------------------------------------------
    calculate scores for cluster labels of different layers using bayesian
    probalblity

    :param im_labels:           labels predicted at every layer
    :param one_hot_label:
    :param clusters_per_layer:  ncluster for kmeans used at every layer
    :return: out_labels, out_preds, beliefs
    ---------------------------------------------------------------------------
    """

    num_layers = len(im_labels)

    if beliefs is None:
        beliefs = []

        for k in range(num_layers-2, -1, -1):
            belief_mat = np.zeros(shape=(clusters_per_layer[k+1],
                                         clusters_per_layer[k]))

            curr_map = im_labels[k].clone()     # curent layer
            prev_map = im_labels[k+1].clone()   # next layer
            b, c, h, w = prev_map.shape
            curr_map = transforms.Resize(size=(h, w),
                                         interpolation=Image.NEAREST)(curr_map)

            curr_map, prev_map = np.uint8(curr_map.cpu().numpy()), \
                                 np.uint8(prev_map.cpu().numpy())

            curr_map = np.squeeze(curr_map)
            prev_map = np.squeeze(prev_map)

            # rprops = regionprops(label_image=prev_map)
            rprops = regionprops(label_image=curr_map)

            for rp in rprops:
                area = rp.area

                lvals = prev_map[curr_map==rp.label]
                # lvals = curr_map[prev_map==rp.label]

                lbls, lfreq = np.unique(lvals, return_counts=True)
                lfreq = lfreq/area

                for l, f in zip(lbls, lfreq):
                    belief_mat[l, rp.label] = f
                    # belief_mat[rp.label, l] = f

            beliefs.append(belief_mat)

        beliefs = [torch.from_numpy(b).to(im_labels[0].device, torch.float)
                   for b in beliefs]

    ob, oc, oh, ow = one_hot_label.shape

    pred_vect = one_hot_label.permute(1, 0, 2, 3).flatten(1)

    out_labels = [im_labels[-1]]
    out_preds  = [one_hot_label]

    for k in range(num_layers-1):
        pred_vect = torch.matmul(torch.transpose(beliefs[k], 0, 1), pred_vect)

        oc = pred_vect.shape[0]

        out_pred_im = pred_vect.reshape(oc,ob,oh,ow).permute(1, 0, 2, 3)
        out_label_im = out_pred_im.max(1)[1]

        out_one_hot = torch.zeros(oc, oh, ow)
        for i in range(oc):
            out_one_hot[i:i+1,:,:][out_label_im==i] = 1

        pred_vect = out_one_hot.flatten(1)

        out_labels.append(out_label_im)
        out_preds.append(out_pred_im)

    # out_labels = [torch.cat((ol, oh), 1)
    #               for ol, oh in zip(out_labels, one_hot_labels)]

    return out_labels, out_preds, beliefs
# -----------------------------------------------------------------------------


def multi_sample_hierarchical_encoding(model,
                                       hfc_predictor,
                                       n_samples,
                                       n_layers):
    """
    ---------------------------------------------------------------------------
    Same as above - belief matrix is calculated offline using a set of
    unlabeled images.

    :param model:           StyleGAN model
    :param hfc_predictor:   trained KMeans models for each layer
    :param n_samples:       number of samples to calculate beliefs form
    :param n_layers:        number of StyleGNA layers to process
    :return:    belief matrix for each layer
    ---------------------------------------------------------------------------
    """

    device = next(model.parameters()).device

    with torch.no_grad():
        beliefs = None

        for curr_s in range(n_samples):

                mean_latent = model.mean_latent(4096)
                truncation  = 0.7

                input_latent = torch.randn(1, 512)
                input_latent = input_latent.to(device)
                orig_img, w_latents = model([input_latent],
                                             return_latents=True,
                                             truncation_latent=mean_latent,
                                             truncation=truncation,
                                             input_is_latent=True)
                test_w_latents = w_latents.detach().clone()

                _, hfeat = \
                    create_images_and_features_from_perturbed_latents(
                        test_w_latents.repeat(1, 1, 1),
                        model,
                        {'truncation': 0.7,
                         'mean_latent': mean_latent},
                        return_feat=True,
                        return_image=True
                    )

                out_preds, out_labels = \
                    hfc_predictor.predict(hfeat[
                                          :n_layers])

                hier_labels, hier_preds, new_belief = \
                    hierarchical_label_encoding(out_labels,
                out_preds[:, -hfc_predictor.clusters_per_layer[-1]:, :, :],
                hfc_predictor.clusters_per_layer,
                beliefs=None)

                if beliefs is None:
                    beliefs = new_belief
                else:
                    beliefs = [0.5*(b1+b2)
                               for b1, b2 in zip(beliefs, new_belief)]

    return beliefs
# -----------------------------------------------------------------------------
