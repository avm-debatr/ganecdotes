from lib.util.util import *

from PIL import Image
from sklearn.cluster import *
from skimage.measure import regionprops
from lib.oneshot.image_augmentor import *

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from baseline.hfc_kmeans.hfc_kmeans_clustering import BaseHFCModel

"""
Older version of Hierarchical KMeans clustering - see hfc_kmeans_clustering to 
use current version.
"""


class HierarchicalKMeansHFC(BaseHFCModel):

    def __init__(self,
                 kmeans_args,
                 base_args):
        """
        -----------------------------------------------------------------------
        Use K-Means hierarchically to perform hidden feature clustering

        :param kmeans_args: sklearn kmenas args
        :param base_args:   constructor args fro BaseHFCModel
        -----------------------------------------------------------------------
        """

        self.kmeans_args = kmeans_args.copy()
        BaseHFCModel.__init__(self, **base_args)
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
        self.clusterers = [None]*self.n_layer

        child_feat = None

        for n in range(self.n_layer-1, -1, -1):
            model, child_feat = self._layerwise_hierarchical_fit(child_feat,
                                                                 hidden_feat[n],
                                                                 n)
            self.clusterers[n] = model
            self.save_model(model, n)

            print(f"Fitted model for Layer {n}")
    # -------------------------------------------------------------------------

    def hierarchical_predict(self, hidden_feat):
        assert len(hidden_feat) == self.n_layer

        cluster_maps = []
        cluster_labels = []

        child_feat = None

        for n in range(self.n_layer-1, -1, -1):
            child_label, child_feat = \
                self._layerwise_hierarchical_predict(child_feat,
                                                     hidden_feat[n],
                                                     n)

            cluster_maps.append(child_feat)
            cluster_labels.append(child_label)

        cluster_maps = torch.cat(cluster_maps[::-1], 1)
        cluster_labels = torch.cat(cluster_labels[::-1], 1)

        return cluster_labels, cluster_maps
    # -------------------------------------------------------------------------

    def _layerwise_hierarchical_fit(self, child_feat, feat, n):
        """
        -----------------------------------------------------------------------
        :param feat:
        :param n:
        :return:
        -----------------------------------------------------------------------
        """
        # initialize current clusterer
        clusterer = KMeans(n_clusters=self.clusters_per_layer[n],
                           **self.kmeans_args)

        # if not the last layer, concatenate the child predictions to
        # current features
        if n!=(self.n_layer-1):

            cb, cc, ch, cw = child_feat.shape
            feat = transforms.Resize(size=(ch, cw),
                                     interpolation=Image.NEAREST)(feat)

            feat = torch.cat((feat.to(DEFAULT_DEVICE),
                              child_feat.to(DEFAULT_DEVICE)), 1)

        b, c, h, w = feat.shape

        # convert to numpy
        hfeat = feat.permute(1, 0, 2, 3).flatten(1).cpu().numpy().T
        clusterer.fit(hfeat)

        # create prediction maps to pass on to next layer
        preds = clusterer.predict(hfeat)
        preds = torch.from_numpy(preds).to(DEFAULT_DEVICE)
        # labels = labels.reshape(h, w, b).permute(2, 0, 1)
        preds = preds.reshape(1, b, h, w).permute(1, 0, 2, 3)

        # convert to one-hot encoding
        pred_maps = torch.zeros(b, self.clusters_per_layer[n], h, w)

        for k in range(self.clusters_per_layer[n]):
            pred_maps[:, k:k+1, :, :][preds == k] = 1

        pred_maps = transforms.Resize(size=(self.out_size,
                                            self.out_size),
                                      interpolation=Image.NEAREST) \
            (pred_maps)

        return clusterer, pred_maps

    # -------------------------------------------------------------------------

    def _layerwise_hierarchical_predict(self,
                                        child_feat,
                                        feat,
                                        n):
        """
        -----------------------------------------------------------------------
        :param feat:
        :param n:
        :param onehot:
        :return:
        -----------------------------------------------------------------------
        """

        # if not the last layer, concatenate the child predictions to
        # current features
        if n != (self.n_layer - 1):
            cb, cc, ch, cw = child_feat.shape
            feat = transforms.Resize(size=(ch, cw),
                                     interpolation=Image.NEAREST)(feat)

            feat = torch.cat((feat.to(DEFAULT_DEVICE),
                              child_feat.to(DEFAULT_DEVICE)), 1)

        b, c, h, w = feat.shape

        # convert to numpy
        hfeat = feat.permute(1, 0, 2, 3).flatten(1).cpu().numpy().T

        clusterer = self.clusterers[n]

        labels = clusterer.predict(hfeat)
        labels = torch.from_numpy(labels)
        # labels = labels.reshape(h, w, b).permute(2, 0, 1)
        labels = labels.reshape(1, b, h, w).permute(1, 0, 2, 3)

        label_maps = torch.zeros(b, self.clusters_per_layer[n], h, w)

        for k in range(self.clusters_per_layer[n]):
            label_maps[:, k:k+1, :, :][labels == k] = 1

        label_maps = transforms.Resize(size=(self.out_size,
                                             self.out_size),
                                       interpolation=Image.NEAREST)(label_maps)
        labels = transforms.Resize(size=(self.out_size,
                                         self.out_size),
                                   interpolation=Image.NEAREST)(labels)

        return labels.unsqueeze(1), label_maps
    # -------------------------------------------------------------------------
