import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib.colors import hsv_to_rgb
from matplotlib.gridspec import GridSpec
from torchvision.utils import make_grid
from PIL import Image
import cv2
import copy
import os
import time


def visualize_label_mask(label, cmap):
    label_image = np.zeros((label.shape[0],
                            label.shape[1],
                            3))

    num_classes = len(cmap)

    for c in range(1, num_classes):
        label_image[label == c] = cmap[c]
    return label_image
# -------------------------------------------------------------------------


class InteractiveLabellerGUI:
    """
    Labelling tool for on-the-fly-segmentation:
    The GUI contains labelling tools developed using
    https://github.com/bryandlee/repurpose-gan

    - to run this gui, use the script run_on_the_fly_segmentor_gui.py

    """

    def __init__(self,
                 one_shot_learner,
                 cmap='jet'):
        """
        -----------------------------------------------------------------------
        :param one_shot_learner: src.OneShotPipeline object
        :param cmap:             colormap for output
        -----------------------------------------------------------------------
        """

        self.one_shot_learner = one_shot_learner
        self.cmap = cmap

        images = self.one_shot_learner.one_shot_img
        images = self.one_shot_learner.transform_im_for_gui(images)
        class_labels = self.one_shot_learner.model_config.classes

        self.num_outs = 8
        self.out_latents = self.one_shot_learner.test_latents[:self.num_outs]
        self.out_grid = self.get_test_image_output(with_labels=False)

        self.snap_dir = os.path.join(self.one_shot_learner.out_dir, 'snaps')
        os.makedirs(self.snap_dir, exist_ok=True)

        self.num_images = len(images)       # number of images 
        self.img_idx = 0                    # index of image being displayed
        self.images = images                # set of images
        
        # initialize all label masks to zero images
        self._reset_label()                 

        # class names 
        if class_labels is not None:
            self.class_labels = class_labels
        else:
            self.class_labels = ['background', 'target']
        
        # assign colors to each class label for drawing labels
        self.num_classes = len(class_labels)
        self.colors = self._sample_colors(self.num_classes)

        self.colors[0] = np.array([1., 1., 1.])
        self._class = 1

        # initialize figure
        self.fig = plt.figure('GANecdotes - One Shot Learning with GANs',
                              figsize=(10,6))
        # self.fig, (self.ax_in, self.ax_out) = plt.subplots(1, 2, figsize=[5, 10])
        # self.ax_in = self.fig.add_subplot(1, 2, 1)
        # self.ax_out = self.fig.add_subplot(1, 2, 2)
        # self.fig = plt.figure(layout='constrained')
        self.fig.tight_layout(pad=0)
        # plt.rcParams['figure.constrained_layout.use'] = True
        # plt.margins(x=0, y=0)

        self.gs = GridSpec(3, 5, figure=self.fig)
        self.ax_in  = self.fig.add_subplot(self.gs[0:2,0:2])
        self.ax_out = self.fig.add_subplot(self.gs[0:, 2:])

        self.fig.subplots_adjust(left=0.0,
                                 bottom=0.0,
                                 right=1.0,
                                 top=1.0,
                                 wspace=0.01)

        self.ax_in.axis('off')
        self.ax_out.axis('off')

        if self.cmap is not None:
            self.ax_img = self.ax_in.imshow(self.images[self.img_idx],
                                         cmap=self.cmap)
            self.ax_img_o = self.ax_out.imshow(self.out_grid,
                                         cmap=self.cmap)

        else:
            self.ax_img = self.ax_in.imshow(self.images[self.img_idx])
            self.ax_img_o = self.ax_out.imshow(self.out_grid)

        # Labelling tools
        self._add_buttons()
        self.fig.canvas.mpl_connect('key_press_event',
                                    self._key_maps)

        self.show_overlay = True
        self.history = []
        self.brush_size = 1

        plt.show()
    # -------------------------------------------------------------------------

    def _sample_colors(self, n=1):
        """
        -----------------------------------------------------------------------
        :param n: number of samples
        :return:
        -----------------------------------------------------------------------
        """
        h = np.linspace(0.0, 1.0, n)[:, np.newaxis]
        s = np.ones((n, 1)) * 0.5
        v = np.ones((n, 1)) * 1.0
        return hsv_to_rgb(np.concatenate([h, s, v], axis=1))
    # -------------------------------------------------------------------------

    def _draw(self, image):
        """
        -----------------------------------------------------------------------
        Draws the image.

        :param image:   image sequence
        :return:
        -----------------------------------------------------------------------
        """
        self.ax_img.set_data(image)
    # -------------------------------------------------------------------------

    def _key_maps(self, event):
        """
        -----------------------------------------------------------------------
        Key maps to functions

        :param event:
        :return:
        -----------------------------------------------------------------------
        """
        key_maps = {
            'c': self._lasso,
            'v': self._poly,
            'z': self._undo,
            'right': self._next_class,
            'left': self._prev_class,
            'o': self._overlay,
            'up': self._brush_up,
            'down': self._brush_down,
        }
        key = event.key.lower()
        if key in key_maps:
            key_maps[key](None)
    # -------------------------------------------------------------------------

    def get_test_image_output(self, with_labels=True):

        out_ims = []

        for i in range(self.num_outs):

            one_shot_latent = self.out_latents[i]
            test_im = self.one_shot_learner.get_image_from_latent(
                                one_shot_latent.unsqueeze(0))
            out_ims.append(test_im[0,:,:,:])

            if with_labels:
                ims, _ = self.one_shot_learner.model(
                    [one_shot_latent],
                    truncation=self.one_shot_learner.model_config.truncation,
                    truncation_latent=self.one_shot_learner.mean_latent,
                    input_is_latent=True,
                    randomize_noise=False
                )

                features, out_labels = \
                    self.one_shot_learner.preprocessor.predict_swav_codes(
                        one_shot_latent)

                # features = features.to('cpu')
                pred = self.one_shot_learner.segmentor(features).to('cpu')
                pred = pred.data.max(1)[1][0,:,:]

            else:
                pred = torch.zeros_like(test_im)[0,0,:,:]

            pred = visualize_label_mask(pred.to('cpu'),
                                        cmap=self.one_shot_learner.color_map)
            pred = 2*pred -1
            pred = torch.from_numpy(pred).permute(2, 0, 1).to(
                                                self.one_shot_learner.device)
            out_ims.append(pred)

        out_grid = make_grid(out_ims, 4)
        out_grid = self.one_shot_learner.transform_im_for_gui(out_grid.unsqueeze(0))[0]

        return out_grid
    # -------------------------------------------------------------------------

    def _add_buttons(self):
        """
        -----------------------------------------------------------------------
        :return:
        -----------------------------------------------------------------------
        """

        # first column - class label, left, right arrows for labels

        # shows the class label being drawn
        # axes_coords = [0.84, 0.94, 0.15, 0.05] # left, bottom, width, height
        axes_coords = [0.00, 0.23, 0.12, 0.04] # left, bottom, width, height

        # class label
        self.class_box = widgets.Button(plt.axes(axes_coords),
                                        self.class_labels[self._class],
                                        color=list(self.colors[self._class]),
                                        hovercolor=list(self.colors[self._class]))

        # left arrow for previous class
        axes_coords_split = copy.deepcopy(axes_coords)
        axes_coords_split = [0.00, 0.18, 0.05, 0.04] # left, bottom, width, height
        self.prev_class = widgets.Button(plt.axes(axes_coords_split), '<')
        self.cid_prev_class = self.prev_class.on_clicked(self._prev_class)

        # right arrow for previous class
        axes_coords_split[0] = 0.07
        self.next_class = widgets.Button(plt.axes(axes_coords_split), '>')
        self.cid_next_class = self.next_class.on_clicked(self._next_class)

        # for lasso style drawing
        axes_coords = [0.00, 0.13, 0.12, 0.04] # left, bottom, width, height
        self.lasso = widgets.Button(plt.axes(axes_coords), 'Lasso (L)')
        self.cid_lasso = self.lasso.on_clicked(self._lasso)

        axes_coords_split = copy.deepcopy(axes_coords)

        # changing brush sizes

        # increase brush size
        axes_coords_split = [0.00, 0.08, 0.05, 0.04] # left, bottom, width, height
        self.brush_up = widgets.Button(plt.axes(axes_coords_split), '+')
        self.cid_brush_up = self.brush_up.on_clicked(self._brush_up)

        # decrease brush size
        axes_coords_split[0] = 0.07
        self.brush_down = widgets.Button(plt.axes(axes_coords_split), '-')
        self.cid_brush_down = self.brush_down.on_clicked(self._brush_down)
        # ---------------------------------------------------------------------

        # Polygon style drawing
        axes_coords = [0.13, 0.23, 0.12, 0.04] # left, bottom, width, height
        self.poly = widgets.Button(plt.axes(axes_coords), 'Polygon (P)')
        self.cid_poly = self.poly.on_clicked(self._poly)

        # undo button
        axes_coords = [0.13, 0.18, 0.12, 0.04] # left, bottom, width, height
        self.undo = widgets.Button(plt.axes(axes_coords), 'Undo (Z)')
        self.cid_undo = self.undo.on_clicked(self._undo)

        # overlays the label mask over the image
        axes_coords = [0.13, 0.13, 0.12, 0.04] # left, bottom, width, height
        self.overlay = widgets.Button(plt.axes(axes_coords), 'Overlay (O)')
        self.cid_overlay = self.overlay.on_clicked(self._overlay)

        # switch between image samples
        axes_coords_split = [0.13, 0.08, 0.05, 0.04] # left, bottom, width, height
        self.prev_img = widgets.Button(plt.axes(axes_coords_split), 'Prev')
        self.cid_prev_img = self.prev_img.on_clicked(self._prev_img)
        axes_coords_split[0] = 0.13+0.07
        self.next_img = widgets.Button(plt.axes(axes_coords_split), 'Next')
        self.cid_next_img = self.next_img.on_clicked(self._next_img)
        # -------------------------------------------------------------------------

        # reset labeller
        axes_coords = [0.27, 0.23, 0.12, 0.04] # left, bottom, width, height
        self.reset = widgets.Button(plt.axes(axes_coords), 'Reset',
                                    color=[1, 0.3, 0.3],
                                    hovercolor=[1, 0.5, 0.5])
        self.cid_reset = self.reset.on_clicked(self._reset)

        # update/train
        axes_coords = [0.27, 0.18, 0.12, 0.04] # left, bottom, width, height
        self.train = widgets.Button(plt.axes(axes_coords), 'Update/Train')
        self.cid_train = self.train.on_clicked(self._update_or_train)

        # regenerate
        axes_coords = [0.27, 0.13, 0.12, 0.04] # left, bottom, width, height
        self.regenerate = widgets.Button(plt.axes(axes_coords), 'Regenerate')
        self.cid_regenerate = self.regenerate.on_clicked(self._regenerate)

        # save output
        axes_coords = [0.27, 0.08, 0.12, 0.04] # left, bottom, width, height
        self.save = widgets.Button(plt.axes(axes_coords), 'Save')
        self.cid_animate = self.save.on_clicked(self._save_output)

        # -------------------------------------------------------------------------

        plt.figtext(0.06, 0.3, 'Labelling Tool for One-Shot Learner',
                    weight='bold')

        axes_coords = [0.0, 0.01, 0.39, 0.04] # left, bottom, width, height
        self.status = widgets.Button(plt.axes(axes_coords),
                                     'Status: Labelling',
                                     color=[0.6, 0.6, 0.6],
                                     hovercolor=[0.6, 0.6, 0.6])
        # self.cid_status = self.reset.on_clicked(self._animate)
    # -------------------------------------------------------------------------

    def _save_output(self, event):

        t0 = int(time.time())
        snap_name = os.path.join(self.snap_dir, f'snap_{t0}.png')

        plt.imsave(snap_name, self.out_grid)

        torch.save(self.out_latents,
                   os.path.join(self.snap_dir, f'latents_{t0}.pt'))
        self.one_shot_learner.logger.info(f"Saved Output + Latents at {t0}")
        return
    # -------------------------------------------------------------------------

    def _regenerate(self, event):

        # new latents
        new_latents = torch.randn_like(self.out_latents)

        for n in range(self.num_outs):
            self.out_latents[n:n+1,:] = self.one_shot_learner.model.style(
                                                            new_latents[n:n+1,:])

        self.one_shot_learner.test_latents = self.out_latents.clone()

        # get out_grid for latents
        self.out_grid = self.get_test_image_output(with_labels=True)

        if self.cmap is not None:
            self.ax_img_o = self.ax_out.imshow(self.out_grid,
                                         cmap=self.cmap)
        else:
            self.ax_img_o = self.ax_out.imshow(self.out_grid)


        return
    # -------------------------------------------------------------------------

    def _update_or_train(self, event):
        self.status.label.set_text("Status: Updating")
        one_shot_label = self.get_labels()
        one_shot_label = torch.from_numpy(one_shot_label).to(
            self.one_shot_learner.one_shot_img.device)

        self.one_shot_learner.one_shot_label = one_shot_label.clone()
        del one_shot_label
        torch.cuda.empty_cache()

        self.one_shot_learner.run_pipeline(blocks_to_run=['train'])
        self.out_grid = self.get_test_image_output(with_labels=True)

        if self.cmap is not None:
            self.ax_img_o = self.ax_out.imshow(self.out_grid,
                                         cmap=self.cmap)
        else:
            self.ax_img_o = self.ax_out.imshow(self.out_grid)

        self.status.label.set_text("Status: Labelling")

        return
    # -------------------------------------------------------------------------

    def _next_class(self, event):
        """
        change class to next level
        :param event:
        :return:
        """
        self._class = (self._class + 1) % self.num_classes
        self._update_class_box()
    # -------------------------------------------------------------------------

    def _prev_class(self, event):
        """
        change class to prev level
        :param event:
        :return:
        """
        self._class = (self._class - 1) % self.num_classes
        self._update_class_box()
    # -------------------------------------------------------------------------

    def _update_class_box(self):
        """
        chnage class box name + color for new label
        :return:
        """
        self.class_box.label.set_text(self.class_labels[self._class])
        self.class_box.color = list(self.colors[self._class])
        self.class_box.hovercolor = self.class_box.color
        self.fig.canvas.draw()
    # -------------------------------------------------------------------------

    def _undo(self, event):
        """
        undo action - this involves chnaguing betwee nhistory states
        :param event:
        :return:
        """
        if len(self.history) > 0:
            self.history.pop(-1)
            self._reset_label(only_current_img=True)
            for inputs in self.history:
                self._update_label(inputs)
            self._draw(self.get_image_label_overlay())
    # -------------------------------------------------------------------------

    def _overlay(self, event):
        """
        show/hide overlay
        :param event:
        :return:
        """
        self.show_overlay = not self.show_overlay
        if self.show_overlay:
            self._draw(self.get_image_label_overlay())
        else:
            self._draw(self.images[self.img_idx])
    # -------------------------------------------------------------------------

    def _reset(self, event):
        """
        reset labels to zero mask images
        :param event:
        :return:
        """
        self.history = []
        self._reset_label(only_current_img=True)
        self._draw(self.images[self.img_idx])
    # -------------------------------------------------------------------------

    def _next_img(self, event):
        """
        switch to next image
        :param event:
        :return:
        """
        self.img_idx = (self.img_idx + 1) % self.num_images
        self._on_img_change()
    # -------------------------------------------------------------------------

    def _prev_img(self, event):
        """
        switch to previous image
        :param event:
        :return:
        """
        self.img_idx = (self.img_idx - 1) % self.num_images
        self._on_img_change()
    # -------------------------------------------------------------------------

    def _on_img_change(self):
        """
        update image being displayed
        :return:
        """
        self.history = []
        self.show_overlay = True
        self._draw(self.get_image_label_overlay())
    # -------------------------------------------------------------------------

    def _poly(self, event):
        """
        ploygon style drawing
        :param event:
        :return:
        """
        self._reset_selectors()
        self.poly_selector = widgets.PolygonSelector(self.ax_in, self._process_polygon)
    # -------------------------------------------------------------------------

    def _process_polygon(self, vert):
        """
        create polgon from selected points
        :param vert:
        :return:
        """
        polygon = np.array(vert, np.int32).reshape((-1, 1, 2))
        inputs = ('poly', polygon, self._class)
        self.history.append(inputs)
        self._update_label(inputs)
        self._after_new_label()
        self._reset_selectors()
    # -------------------------------------------------------------------------

    def _brush_up(self, dummy):
        """
        brush size++
        :param dummy:
        :return:
        """
        self.brush_size += 1
    # -------------------------------------------------------------------------

    def _brush_down(self, dummy):
        """
        brush size --
        :param dummy:
        :return:
        """
        self.brush_size = max(self.brush_size - 1, 1)
    # -------------------------------------------------------------------------

    def _lasso(self, event):
        """
        lasso style drawing
        :param event:
        :return:
        """
        self._reset_selectors()
        self.lasso_selector = widgets.LassoSelector(
            self.ax_in, self._process_lasso, lineprops=dict(linewidth=self.brush_size // 2)
        )
    # -------------------------------------------------------------------------

    def _process_lasso(self, vert):
        """

        :param vert:
        :return:
        """
        path = np.array(vert, np.int32).reshape((-1, 1, 2))
        path = np.unique(path, axis=1)
        inputs = ('lasso', path, self._class, self.brush_size)
        self.history.append(inputs)
        self._update_label(inputs)
        self._after_new_label()
        self._reset_selectors()
    # -------------------------------------------------------------------------

    def _reset_selectors(self):
        """

        :return:
        """
        if hasattr(self, 'lasso_selector'):
            self.lasso_selector.set_visible(False)
            del (self.lasso_selector)
        if hasattr(self, 'poly_selector'):
            self.poly_selector.set_visible(False)
            del (self.poly_selector)
    # -------------------------------------------------------------------------

    def _after_new_label(self):
        self.show_overlay = True
        self._draw(self.get_image_label_overlay())
    # -------------------------------------------------------------------------

    def _reset_label(self, only_current_img=False):
        if only_current_img:
            self.labels[self.img_idx] = np.zeros(
                (self.images.shape[1], self.images.shape[2]),
                np.uint8
            )
        else:
            self.labels = np.zeros(
                (self.num_images, self.images.shape[1], self.images.shape[2]),
                np.uint8
            )
    # -------------------------------------------------------------------------

    def _update_label(self, inputs):
        if inputs[0] == 'poly':
            self.labels[self.img_idx] = cv2.fillPoly(
                self.labels[self.img_idx], [inputs[1]], inputs[2], 0
            )
        elif inputs[0] == 'lasso':
            self.labels[self.img_idx] = cv2.polylines(
                self.labels[self.img_idx], [inputs[1]], isClosed=False,
                color=inputs[2], thickness=inputs[3]
            )
    # -------------------------------------------------------------------------

    def get_image_label_overlay(self):
        overlay = self.images[self.img_idx].copy()
        label_image = self.get_visualized_label()
        non_zeros = label_image > 0
        overlay[non_zeros] = label_image[non_zeros]
        return overlay
    # -------------------------------------------------------------------------

    def get_visualized_label(self, label=None):
        if label is None:
            label = self.labels[self.img_idx]

        label_image = np.zeros_like(self.images[self.img_idx])
        for c in range(1, self.num_classes):
            label_image[label == c] = self.colors[c]
        return label_image
    # -------------------------------------------------------------------------

    def get_labels(self):
        return self.labels
    # -------------------------------------------------------------------------


def sample_label_colors(n=1):
    """
    -----------------------------------------------------------------------
    :param n: number of samples
    :return:
    -----------------------------------------------------------------------
    """
    h = np.linspace(0.0, 1.0, n)[:, np.newaxis]
    # h2 = np.copy(h)
    # h[::2, :] = h2[1::2, :]
    # h[1::2, :] = h2[::2, :]

    s = np.ones((n, 1)) * 0.5
    v = np.ones((n, 1)) * 1.0
    return hsv_to_rgb(np.concatenate([h, s, v], axis=1))
# -------------------------------------------------------------------------


if __name__=="__main__":
    from models.stylegan2.model import Generator
    import torch

    device = 'cuda:0'
    image_size = 256
    n_samples = 1

    generator_path = '/mnt/cloudNAS3/Ankit/1_Datasets/stylegan_pretrained/stylegan2-cat-config-f.pt'

    latent_dim, truncation, imshow_size = 512, 0.7, 3

    generator = Generator(image_size, latent_dim, 8)
    generator_ckpt = torch.load(generator_path, map_location='cpu')
    generator.load_state_dict(generator_ckpt["g_ema"], strict=False)
    generator.eval().to(device)

    with torch.no_grad():
        trunc_mean = generator.mean_latent(4096).detach().clone()
        latent = generator.get_latent(torch.randn(n_samples, latent_dim, device=device))
        img_gen, features = generator([latent],
                                       truncation=truncation,
                                       truncation_latent=trunc_mean,
                                       input_is_latent=True,
                                       randomize_noise=True)
        # img_gen, _, out = generator([latent],
        #                             input_is_latent=True,
        #                             randomize_noise=False,
        #                             return_features=8)

        torch.cuda.empty_cache()

    class_labels = ['background', 'fur', 'whisker', 'eye', 'nose', 'mouth', 'ear']
    labeller = InteractiveLabellerGUI(
                    img_gen.clamp_(-1., 1.).detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5,
                    class_labels)
