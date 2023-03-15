import torch, os, time
from collections import OrderedDict

from models.baggan.gan_util import *
from lib.util.util import get_logger


class GANBaseModel(object):

    def __init__(self, config):
        """
        -----------------------------------------------------------------------
        Initializing the Base Class for BagGAN - different versions have
        different architectures.

        Adopted from the pix2pix architecture.

        :param config: Options object
        -----------------------------------------------------------------------
        """

        # Options Object containing parameters for the class + GAN ------------
        self.config   = config
        self.gpu_ids  = self.config.gpu_ids
        self.is_train = self.config.is_train
        # ---------------------------------------------------------------------

        # Generate the logger -------------------------------------------------
        if config.training_log_path is None:
            config.training_log_path = os.path.join(
                config.expt_dir,
                'expt_%s'%config.expt,
                time.strftime('baggan_training_%m%d%Y_%H%M%S.log',
                              time.localtime())
            )

        if not os.path.exists(config.training_log_path):
            lf = open(config.training_log_path, 'w+')
            lf.write("============ BagGAN TRAINING =============\n")
            lf.close()
        else:
            pass
            # open(configs.training_log_path, 'w').close()

        self.logger = get_logger(config.baggan_logger_name,
                                 config.training_log_path)
        # ---------------------------------------------------------------------

        # get device name: CPU or GPU
        self.device = torch.device('cuda:%d'%self.gpu_ids[0]) \
                      if self.gpu_ids else torch.device('cpu')

        # save all the checkpoints to save_dir
        self.save_dir = config.checkpoint_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Parameter that needs to be specified - they are empty for the base
        # model but must be initialized by the child classes

        self.loss_names = []
        self.model_names = dict()
        self.optimizers = []

        self.metric = 0  # used for learning rate policy 'plateau'
        # ---------------------------------------------------------------------

    def setup_gan(self):
        """
        -----------------------------------------------------------------------
        Set up the GAN models. optimizers and losses in the BagGAN

        :return:
        -----------------------------------------------------------------------
        """

        # Loading Learning Rate Schedulers - only done for training -----------
        if self.is_train:
            self.schedulers = [get_scheduler(optimizer,
                                             self.config.lr_policy,
                                             **self.config.lr_params)
                               for optimizer in self.optimizers]
        # ---------------------------------------------------------------------

        # If testing or resuming the training of a saved expt, the models need
        # to be loaded
        if self.config.continue_train:
            # determines the file to be loaded
            load_suffix = 'e_%d_i_%d' % (self.config.load_epoch, 0) \
                           if self.config.load_epoch is not None else 'final'
            self.load_networks(load_suffix)

        if not self.is_train:

            # determines the file to be loaded
            load_suffix = 'e_%d_i_%d' % (self.config.load_epoch, 0) \
                           if self.config.load_epoch is not None else 'final'
            self.load_networks(load_suffix)

        self.print_networks(self.config.verbose)
    # -------------------------------------------------------------------------

    def eval(self):
        """
        -----------------------------------------------------------------------
        Activate eval mode in the model for testing.

        -----------------------------------------------------------------------
        """

        for name in self.model_names.keys():
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()
    # -------------------------------------------------------------------------

    def update_learning_rate(self):
        """
        -----------------------------------------------------------------------
        Update learning rates for all the algos;
        called at the end of every epoch

        -----------------------------------------------------------------------
        """

        for scheduler in self.schedulers:
            if self.config.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        self.logger.info('*'*30+' Updated learning rate to %.7f ***' % lr+ '*'*30)
    # -------------------------------------------------------------------------

    def get_current_losses(self):
        """
        -----------------------------------------------------------------------
        Return training losses / errors. train_v4.py will print out these errors
        on console, and save them to a file

        -----------------------------------------------------------------------
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
                # float(...) works for both scalar tensor and float number
        return errors_ret
    # -------------------------------------------------------------------------

    def save_networks(self, suffix):
        """
        -----------------------------------------------------------------------
        Save all the algos to the disk.

        Parameters:
        * epoch (int) -- current epoch; used in the file name
                         '%s_net_%s.pth' % (epoch, name)

        -----------------------------------------------------------------------
        """
        for name in self.model_names.keys():
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (self.model_names[name],
                                                   suffix)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
    # -------------------------------------------------------------------------

    def load_networks(self, load_suffix):
        """
        -----------------------------------------------------------------------
        Load all the algos from the disk.

        Parameters:
        * epoch (int) - current epoch;
                        used in the file name '%s_net_%s.pth' % (epoch, name)
        -----------------------------------------------------------------------
        """
        print(load_suffix)

        # Load the saved nets for both the generator and discriminator
        for name in self.model_names.keys():

            # load the filename for the saved net -----------------------------
            assert isinstance(name, str)
            load_filename = '%s_net_%s.pth' % (self.model_names[name],
                                               load_suffix)
            load_path = os.path.join(self.save_dir, load_filename)

            if not os.path.exists(load_path):
                raise FileNotFoundError('Could not find model in the specified '
                                        'path! Make sure the right epoch number '
                                        'is specified in the config file.')

            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            # -----------------------------------------------------------------

            self.logger.info('Loading the model from %s' % load_path)

            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device

            state_dict = torch.load(load_path,
                                    map_location=self.device)

            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.load_state_dict(state_dict)

            self.logger.info('Model for %s loaded'%name)
    # -------------------------------------------------------------------------

    def load_networks_raw(self, epoch, itern):
        """
        -----------------------------------------------------------------------
        Load all the algos from the disk.

        Parameters:
        * epoch (int) - current epoch;
                        used in the file name '%s_net_%s.pth' % (epoch, name)
        -----------------------------------------------------------------------
        """
        # Load the saved nets for both the generator and discriminator
        for name in self.model_names.keys():

            # load the filename for the saved net -----------------------------
            assert isinstance(name, str)
            load_filename = '%s_net_e_%s_i_%i.pth' % (self.model_names[name],
                                                      epoch,
                                                      itern)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            # -----------------------------------------------------------------

            self.logger.info('Loading the model from %s' % load_path)

            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device

            net = torch.load(load_path,
                                    map_location=self.device)
            net.eval()

            self.logger.info('Model for %s loaded'%name)
    # -------------------------------------------------------------------------

    def print_networks(self, verbose):
        """
        -----------------------------------------------------------------------
        Print the total number of parameters in the network and (if verbose)
        network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        -----------------------------------------------------------------------
        """
        self.logger.info('-'*20 + ' Networks initialized '+'-'*20)
        for name in self.model_names.keys():

            assert  isinstance(name, str)
            net = getattr(self, name)
            num_params = 0

            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                self.logger.info('\n'+net.__str__())

            self.logger.info('[Network %s] Total number of parameters : '
                  '%.3f M' % (self.model_names[name],
                              num_params / 1e6))
        self.logger.info('-'*80)
    # -------------------------------------------------------------------------

    def set_requires_grad(self, nets, requires_grad=False):
        """
        -----------------------------------------------------------------------
        Set requires_grad=False for all the algos to avoid unnecessary
        computations

        Parameters:
        * nets (network list)  - a list of algos
        * requires_grad (bool) - whether the algos require gradients or not
        -----------------------------------------------------------------------
        """

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    # -------------------------------------------------------------------------

