from lib.util.util import *
from PIL import Image

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

import imageio
from skimage.transform import rescale


def create_pil_collage(images,
                       fpath=None,
                       layout=None,
                       vlims=None,
                       return_im=False):
    """
    ---------------------------------------------------------------------------
    :param images:
    :param fpath:
    :param layout:
    :param vlims:
    :return:
    ---------------------------------------------------------------------------
    """

    if layout is None: layout = (len(images), 1)

    if vlims is not None:
        images = [np.clip(x, vlims[0], vlims[1])/vlims[1]*255
                  for x in images]
    else:
        images = [x/x.max()*255 for x in images]

    assert layout[0]*layout[1] == len(images)

    rows, cols = layout

    # if len(images[0].shape)==2:
    #     ra, ca = 0, 1
    # if len(images[0].shape) == 3:
    #     ra, ca = 1, 2
    ra, ca = 0, 1

    collage = np.concatenate([np.concatenate(images[x*cols:(x+1)*cols],
                                             axis=ca)
                        for x in range(rows)],
                       axis=ra)

    if len(images[0].shape)==2:
        im = Image.fromarray(collage)
        im = im.convert('L')
    elif len(images[0].shape) == 3:
        im = Image.fromarray(np.uint8(collage))

    # im = Image.fromarray(np.clip(orig_slice[::-1, :] + 1000, 0, 2000)
    #                      / 2000 * 255)
    im = im.convert('RGB')

    if fpath is not None:
        im.save(fpath)

    if return_im:
        return im
# -----------------------------------------------------------------------------


def quick_imshow(nrows, ncols=1,
                 images=None,
                 titles=None,
                 colorbar=True,
                 vmax=None,
                 vmin=None,
                 figsize=None,
                 figtitle=None,
                 visibleaxis=False,
                 colormap='jet',
                 saveas=''):
    """-------------------------------------------------------------------------
    Convenience function that make subplots of imshow

    :param  nrows - number of rows
    :param  ncols - number of cols
    :param  images - list of images
    :param  titles - list of titles
    :param  vmax - tuple of vmax for the colormap. If scalar, the same value is
                   used for all subplots. If one of the entries is None, no
                   colormap for that subplot will be drawn.
    :param  vmin - tuple of vmin

    :return: f - the figure handle
             axes - axes or array of axes objects
             caxes - tuple of axes image
    -------------------------------------------------------------------------"""
    if isinstance(nrows, ndarray):
        images = nrows
        nrows = 1
        ncols = 1

    if figsize == None:
        # 1.0 translates to 100 pixels of the figure
        s = 3.5
        if figtitle:
            figsize = (s * ncols, s * nrows + 0.5)
        else:
            figsize = (s * ncols, s * nrows)

    if nrows == ncols == 1:
        f, ax = subplots(figsize=figsize)
        cax = ax.imshow(images, cmap=colormap, vmax=vmax, vmin=vmin)
        if colorbar:
            f.colorbar(cax, ax=ax)
        if titles != None:
            ax.set_title(titles)
        if figtitle != None:
            f.suptitle(figtitle)
        cax.axes.get_xaxis().set_visible(visibleaxis)
        cax.axes.get_yaxis().set_visible(visibleaxis)
        return f, ax, cax

    f, axes = subplots(nrows, ncols, figsize=figsize)
    caxes = []
    i = 0
    for ax, img in zip(axes.flat, images):
        if isinstance(vmax, tuple) and isinstance(vmin, tuple):
            if vmax[i] is not None and vmin[i] is not None:
                cax = ax.imshow(img, cmap=colormap, vmax=vmax[i], vmin=vmin[i])
            else:
                cax = ax.imshow(img, cmap=colormap)
        elif isinstance(vmax, tuple) and vmin is None:
            if vmax[i] is not None:
                cax = ax.imshow(img, cmap=colormap, vmax=vmax[i], vmin=0)
            else:
                cax = ax.imshow(img, cmap=colormap)
        elif vmax is None and vmin is None:
            cax = ax.imshow(img, cmap=colormap)
        else:
            cax = ax.imshow(img, cmap=colormap, vmax=vmax, vmin=vmin)
        if titles != None:
            ax.set_title(titles[i])
        if colorbar:
            f.colorbar(cax, ax=ax)
        caxes.append(cax)
        cax.axes.get_xaxis().set_visible(visibleaxis)
        cax.axes.get_yaxis().set_visible(visibleaxis)
        i = i + 1
    if figtitle != None:
        f.suptitle(figtitle)
    if saveas != '':
        f.savefig(saveas)
    return f, axes, tuple(caxes)
# -----------------------------------------------------------------------------


def slide_show(image, dt=0.01, vmax=None, vmin=None):
    """
    ---------------------------------------------------------------------------
    Slide show for visualizing an image volume. Image is (w, h, d)

    :param image: (w, h, d), slides are 2D images along the depth axis
    :param dt:      transition time
    :param vmax:    maximum cliiping value
    :param vmin:    minimum clipping value
    :return:
    ---------------------------------------------------------------------------
    """

    if image.dtype == bool:
        image *= 1.0
    if vmax is None:
        vmax = image.max()
    if vmin is None:
        vmin = image.min()
    plt.ion()
    plt.figure()
    for i in range(image.shape[2]):
        plt.cla()
        cax = plt.imshow(image[:, :, i], cmap='jet', vmin=vmin, vmax=vmax)
        plt.title(str('Slice: %i' % i))
        if i == 0:
            cf = plt.gcf()
            ca = plt.gca()
            cf.colorbar(cax, ax=ca)
        plt.pause(dt)
        plt.draw()
# -----------------------------------------------------------------------------


def plot_boxplot(fname,
                 vectors,
                 titles=None,
                 lbl_rotation=None):
    """
    ---------------------------------------------------------------------------
    Create box plot for a set of vectors

    :param fname:        filepath
    :param vectors:      data for plotting boxplots must contains two list:
                         vectors[0] : labels for boxplots
                         vectors[1] : data
    :param titles:       dict for xlabel, ylabel, title
    :return:
    ---------------------------------------------------------------------------
    """

    assert len(vectors[0])==len(vectors[1])

    plt.figure()
    plt.boxplot(vectors[1], showfliers=False)
    plt.xticks(len(vectors[0]),
               vectors[0],
               rotation='vertical' if lbl_rotation is None else lbl_rotation)

    if titles is not None:
        plt.xlabel(titles['xlabel'])
        plt.ylabel(titles['ylabel'])
        plt.title(titles['title'])

    plt.savefig(fname)
    plt.tight_layout()
    plt.close()
# -----------------------------------------------------------------------------


def plot_histogram_1d(fname,
                      vectors,
                      titles=None,
                      legend=True,
                      is_hist=True,
                      hist_params=None,
                      ):
    """
    ---------------------------------------------------------------------------
    Create box plot for a set of vectors

    :param fname:        filepath
    :param vectors:      data for plotting boxplots must contains two list:
                         vectors[0] : labels for boxplots
                         vectors[1] : data
    :param titles:       dict for xlabel, ylabel, title
    :return:
    ---------------------------------------------------------------------------
    """

    assert len(vectors[0]) == len(vectors[1])

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    if not is_hist:
        vectors = [np.histogram(v, **hist_params) for v in vectors]

    ax.plot(vectors[1], label=vectors[0])

    if legend: plt.legend()

    if titles is not None:
        plt.xlabel(titles['xlabel'])
        plt.ylabel(titles['ylabel'])
        plt.title(titles['title'])

    plt.savefig(fname)
    plt.tight_layout()
    plt.close()
# ------------------------------------------------------------------------------


def load_image(im_path):

    ext = os.path.splitext(im_path)[-1]

    assert ext in ['.png', '.jpg', '.tiff', '.npz',
                   '.npy', '.gz', '.dcs', '.dcm'], 'Format not supported!'

    if ext in ['.png', '.jpg', '.tiff', '.npz']:
        return imread(im_path)
    elif ext in ['.npz', '.npy']:
        return load(im_path)['arr_0']
    elif ext=='.gz':
        assert im_path[:-5]=='.fits', 'Not a FITS file!'
        return read_fits_data(im_path)
    # -------------------------------------------------------------------------


def create_gif(fname, input_im, stride=1, scale=None, fps=5):
    """
    ---------------------------------------------------------------------------
    Create a GIF from a list of images, image paths or a 3D numpy array

    :param fname:
    :param input_im:
    :param stride:
    :return:
    ---------------------------------------------------------------------------
    """
    
    if isinstance(input_im, list):
        if isinstance(input_im, str):
            input_im = np.dstack(tuple([load_image(x) for x in input_im]))
        elif isinstance(input_im, np.ndarray):
            input_im = np.dstack(tuple([x for x in input_im]))

    elif isinstance(input_im, np.ndarray):
        pass

    if input_im.dtype != uint8:
        input_im = (input_im-input_im.min())/(input_im.max()-input_im.min())
        input_im = (input_im*255).astype(uint8)

    if scale is None:
        # imageio.mimsave(fname,
        #                 [input_im[:,:, z]
        #                  for z in range(0,input_im.shape[2], stride)],
        #                 fps=fps)
        if len(input_im.shape)==3:
            imageio.mimsave(fname,
                            [input_im[:,:, z]
                             for z in range(0,input_im.shape[2], stride)],
                            fps=fps)
        elif len(input_im.shape)==4:
            imageio.mimsave(fname,
                            [input_im[z, :,:,:]
                             for z in range(0,input_im.shape[3], stride)],
                            fps=fps)
    else:

        if len(input_im.shape)==3:
            imageio.mimsave(fname,
                            [rescale(input_im[:,:, z],
                                     scale=scale,
                                     preserve_range=True)
                             for z in range(0,input_im.shape[2], stride)],
                            fps=fps)
        elif len(input_im.shape)==4:
            imageio.mimsave(fname,
                            [rescale(input_im[z, :,:,:],
                                     scale=scale,
                                     preserve_range=True)
                             for z in range(0,input_im.shape[2], stride)],
                            fps=fps)

    # -------------------------------------------------------------------------


def plot_image_on_axis(ax,
                       fig,
                       im=None,
                       title=None,
                       show_axes=False,
                       # show_cbar=False,
                       im_args={'cmap': 'gray'},
                       title_args={'fontsize': 12}
                       ):
    """
    ---------------------------------------------------------------------------
    Plot an image on an axis within a matplotlib figure.

    :param ax:          Axes object
    :param fig:         Figure Object
    :param im:          2D np.array for the image
    :param title:       Title for the plot
    :param show_axes:   if axes are to be shown
    :param show_cbar:   if colorbar is to be shown
    :param im_args:     arguments for displaying image
    :param title_args:  arguments for displaying title

    :return:
    ---------------------------------------------------------------------------
    """

    if not show_axes:
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

    im_ax = None

    # if show_cbar:           fig.colorbar(ax=ax, cax=ax)
    if im is not None:      im_ax = ax.imshow(im, **im_args)
    if title is not None:   ax.set_title(title, **title_args)
    return im_ax
    # -------------------------------------------------------------------------
