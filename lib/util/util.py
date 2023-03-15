from pylab import *
from numpy import *
from astropy.io import fits as pyfits
import pickle
from matplotlib.patches import Ellipse
import torch
import torchvision.transforms as transforms
import importlib.util as imp_util
from lib.__init__ import *
import warnings
import argparse
import sys

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import copy

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(cfg_path, cfg_name='name'):
    """
    ---------------------------------------------------------------------------


    :param cfg_path:
    :return:
    ---------------------------------------------------------------------------
    """
    assert os.path.exists(cfg_path), "Config file not found!"

    config = imp_util.spec_from_file_location(cfg_name, cfg_path)
    config_module = imp_util.module_from_spec(config)
    config.loader.exec_module(config_module)

    return config_module
# -----------------------------------------------------------------------------


def get_logger(lname, logfile):
    """
    ---------------------------------------------------------------------------
    Function to create a python logger for a given class.

    :param lname:       name for logger - printed on terminal
    :param logfile:     log file path
    :return:
    ---------------------------------------------------------------------------
    """
    # print("="*160)
    # print("="*160)
    # print("Called for :", lname)
    # print("="*160)
    # print("="*160)

    # Create logger object
    logger = logging.getLogger(lname)
    logger.setLevel(logging.INFO)

    # Create file handler
    f_handler = logging.FileHandler(logfile, mode='a')
    s_handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    formatter = logging.Formatter(
                        '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s')
    f_handler.setFormatter(formatter)
    s_handler.setFormatter(formatter)

    logger.addHandler(f_handler)
    logger.addHandler(s_handler)

    return logger
# -----------------------------------------------------------------------------


class ConfigLoader(object):
    """
    ---------------------------------------------------------------------------
    Class for loading a config file for GAN using commandline arguments
    ---------------------------------------------------------------------------
    """

    def __init__(self, default_config):
        """
        -----------------------------------------------------------------------

        :param default_config:
        -----------------------------------------------------------------------
        """

        self.default_config = default_config
        self.config = default_config

        parser = argparse.ArgumentParser(description="BagGAN Configuration Loader")

        parser.add_argument('--config',
                            help='Path to the config file. Parameters '
                                 'within the config file can be fed as '
                                 'arguments to the script',
                            default=None
                            )

        for attr in [x for x in dir(self.config)
                       if x[0]!='_' and
                          not isinstance(self.config.__dict__[x], dict) and
                          x not in ['os', 'time']]:
            parser.add_argument('--'+attr)

        args = parser.parse_args()

        if args.config is not None:

            assert os.path.exists(args.config), "Config file not found!"

            self.config = imp_util.spec_from_file_location("config", args.config)
            self.config_module = imp_util.module_from_spec(self.config)
            self.config.loader.exec_module(self.config_module)

        for attr, val in vars(args).items():
            self.config.__dict__[attr] = val
    # -------------------------------------------------------------------------

    def __call__(self, default=False):
        return self.config if default else self.default_config
# -----------------------------------------------------------------------------


def read_fits_data(input_file_name, field=1):
    """
    ---------------------------------------------------------------------------
    Loads a FITS image file

    :param input_file_name - file path
    :return image as a numpy ndarray
    ---------------------------------------------------------------------------
    """

    return  pyfits.open(input_file_name,
                        ignore_missing_end=True)[field].data
# -----------------------------------------------------------------------------


def save_fits_data(file_path, out_image):
    """
    ---------------------------------------------------------------------------
    Save an image as a FITS file

    :param file_path:   path to the fits file
    :param out_image:   output image to be saved
    :return:
    ---------------------------------------------------------------------------
    """

    if os.path.exists(file_path):
        os.remove(file_path)

    imheader = pyfits.Header()
    hdu_list = pyfits.CompImageHDU(out_image, imheader)
    hdu_list.writeto(file_path)
# -----------------------------------------------------------------------------

def scatter_ellipse(X, labels, mu, R, figsize=(5, 5), s=0.01, alpha=0.1):
    """
    ---------------------------------------------------------------------------
    2D scatter plot with ellipse drawn based on mean and covariance.

    :param X: samples, (N, 2)
    :param labels: integer labels, (N,)
    :param mu: centroids, (k, 2)
    :param R: covariances, (k, 2, 2)
    :return:
    ---------------------------------------------------------------------------
    """
    k = len(unique(labels))

    f, ax = subplots(figsize=figsize)
    ax.scatter(X[:, 0], X[:, 1],
               s=s, c=labels, alpha=alpha, cmap='jet')

    for m in range(k):
        vals, vecs = eigh(R[m])
        x, y = vecs[:, 0]
        w, h = 2 * sqrt(vals)
        theta = degrees(arctan2(y, x))
        ax.add_artist(
            Ellipse(xy=mu[m], width=w, height=h, angle=theta, fill=False,
                    edgecolor='r'))

    return f, ax
# -----------------------------------------------------------------------------

def read_sl_metadata_file(sample_path):
    """
    -----------------------------------------------------------------------
    Read the sl_metadata.pyc file for the given sample

    :param sample_path: name of sample directory
    :return:
    -----------------------------------------------------------------------
    """

    # fpath = self.f_loc['sl_metadata'] % sample
    fpath = sample_path

    with open(fpath, 'rb') as f:
        sl_data = pickle.load(f, encoding='latin1')
        f.close()

    return sl_data
# -------------------------------------------------------------------------


def send_email_notification(body,
                            pswd,
                            receiver="ankit.31457@gmail.com",
                            sender="hermian.naga.20@gmail.com",
                            subject="Email Auto-alert"):
    """
    ---------------------------------------------------------------------------

    :param body:
    :param receiver:
    :param sender:
    :param subject:
    :return:
    ---------------------------------------------------------------------------
    """
    sender_port = 465
    sender_email = sender
    sender_password = pswd

    msg = MIMEMultipart()
    msg['From'] = sender    if sender is None else sender_email
    msg['To']   = receiver

    default_subject = "DEBISim Code Alert: " + \
                      time.strftime('%m-%d-%Y %H:%M:%S', time.localtime())
    msg['Subject']  = default_subject if subject is None else subject

    msg.attach(MIMEText(body))

    mail_server = smtplib.SMTP_SSL('smtp.gmail.com', sender_port)
    mail_server.ehlo()
    mail_server.login(sender_email, sender_password)
    mail_server.sendmail(sender    if sender is None else sender_email,
                         receiver,
                         msg.as_string())
    mail_server.close()
# -----------------------------------------------------------------------------
