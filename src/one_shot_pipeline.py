from lib.util.util import *
from configs import mapper as config_mapper

import baseline.repurposegan.base      as repurposegan
import baseline.datasetgan.base        as datasetgan
import baseline.hfc_kmeans.base        as hfc_kmeans
import baseline.hfc_with_simclr.base   as hfc_with_simclr
import hfc_with_swav.base              as hfc_with_swav

from lib.gui.labeller import OneShotLabellerGUI, sample_label_colors, \
                             visualize_label_mask

from lib.metrics.segmentation import *
from lib.util.visualization import *
import pandas as pd

from tqdm import tqdm

from models.stylegan2.model import Generator
from models.baggan.bagganhq import BagGANHQ


"""
-------------------------------------------------------------------------------
- Implements a pipeline to perform one-shot learning for automatic segmentation 
of synthetic images generated using StyleGANs.
-------------------------------------------------------------------------------
"""

MAX_TEST_BATCH = 1


class OneShotPipeline(object):

    def __init__(self,
                 out_dir,
                 exp_name='',
                 model='ffhq-256',
                 segmentor='hfc_kmeans',
                 trainer='supervised',
                 tester='all',
                 mode='offline',
                 inputs='saved',
                 custom=None,
                 device='cuda:0',
                 num_test_samples=None):
        """
        -----------------------------------------------------------------------
        Establishes a pipeline for one-shot learning both for online mode
        where the user manually labels a sample using a GUI or offline from
        a pre-saved one shot sample.

        :param model:
        :param segmentor:
        :param trainer:
        :param mode:
        :param custom:
        -----------------------------------------------------------------------
        """
        # make output directory for storing results
        self.out_dir = out_dir

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        # logfile for storing output
        self.start_time = time.strftime("%m%d%Y_%H%M%S", time.localtime())
        self.logfile = os.path.join(self.out_dir,
                                    f'one_shot_learner_{self.start_time}.log')
        self.logger = get_logger('OneShot', self.logfile)

        # Tensorboard summary
        self.summary_writer = SummaryWriter(
                    log_dir=os.path.join(self.out_dir,
                                         'tensorboard',
                                         f'run_{self.start_time}'))

        # parameter names
        self.model_str = model
        self.seg_str   = segmentor
        self.train_str = trainer
        self.test_str  = tester
        self.mode      = mode
        self.inputs    = inputs
        self.device    = device
        self.exp_name  = exp_name

        self.logger.info("="*80)
        self.logger.info("One-Shot Learning Pipeline for StyleGANs")
        self.logger.info("="*80+'\n')

        # get config file locations for different blocks
        self.logger.info("Loading Configurations ....")
        self.logger.info(self.exp_name)

        self.configs = dict()
        self.configs['model'] = config_mapper.models[self.model_str]
        self.configs['seg'] = config_mapper.segmentors[self.seg_str]
        self.configs['trainer'] = config_mapper.trainer[self.train_str]
        # self.configs['tester'] = config_mapper['model'][self.test_str]

        # change config file paths if custom file is given
        if custom is not None:
            self.configs.update(custom)

        # load all blocks
        self.logger.info("Loading Pipeline Blocks ...\n")
        self.load_model()
        self.load_segmentor()
        self.load_trainer()

        self.logger.info("Loading Pipeline Blocks ... Done.")

        self.num_test_samples = num_test_samples

    # -------------------------------------------------------------------------

    def load_model(self):
        """
        -----------------------------------------------------------------------
        loads pre-trained StyleGAN2 Model for image generation

        :return:
        -----------------------------------------------------------------------
        """
        self.logger.info("*"*20)

        self.logger.info("Loading Pretrained StyleGAN2 Model ... ")

        # loading config file for model
        self.model_config = load_config(self.configs['model'],
                                        'model_config')

        # load pretrained StyleGAN model
        if not self.model_config.is_baggan:

            # if it is a standard pretrained model
            self.model = Generator(**self.model_config.gen_args)

            g_state_dict = torch.load(self.model_config.model_path,
                                      map_location=self.device)
            if 'g_ema' in g_state_dict.keys():
                self.model.load_state_dict(g_state_dict["g_ema"],
                                           strict=False)
            else:
                self.model.load_state_dict(g_state_dict,
                                           strict=False)

        else:
            # if it is a BagGAN model
            baggan = BagGANHQ(load_config(self.model_config.config_path,
                                          'baggan_config'))
            baggan.setup_gan()
            self.model = baggan.generator.module

        self.logger.info("Done")

        self.logger.info('')
        self.logger.info("-"*40)
        self.logger.info(f"Model Name: {self.model_str}")
        if not self.model_config.is_baggan:
            self.logger.info(f"Model Path: {self.model_config.model_path}")
        self.logger.info(f"BagGAN? {self.model_config.is_baggan}")
        self.logger.info('')
        self.logger.info(f"Model Args - ")
        for k, v in self.model_config.gen_args.items():
            if not isinstance(v, list):
                self.logger.info(f"{k}: {v}")

        self.logger.info("-"*40+'\n')

        self.color_map = sample_label_colors(len(self.model_config.classes))

        # initialize model
        self.model.eval()
        self.model = self.model.cuda()
        self.mean_latent = self.model.mean_latent(
                                self.model_config.num_latents_for_mean)
    # -------------------------------------------------------------------------

    def load_segmentor(self):
        """
        -----------------------------------------------------------------------
        Loads the segmentation network for one-shot learning

        :return:
        -----------------------------------------------------------------------
        """
        self.logger.info("")
        self.logger.info("Loading Segmentor Network ... ")

        self.seg_config = load_config(self.configs['seg'],
                                      'seg_config')

        if self.seg_str=='repurposegan':
            self.segmentor = repurposegan.segmentor
            self.preprocessor = repurposegan.preprocess

        elif self.seg_str=='datasetgan':
            self.segmentor = datasetgan.segmentor
            self.preprocessor = datasetgan.preprocess

        elif self.seg_str.find('hfc_with_swav')!=-1:
            self.segmentor = hfc_with_swav.segmentor
            self.preprocessor = hfc_with_swav.preprocessor

        elif self.seg_str=='hfc_with_simclr':
            self.segmentor = hfc_with_simclr.segmentor
            self.preprocessor = hfc_with_simclr.preprocessor

        elif self.seg_str=='hfc_kmeans':
            self.segmentor = hfc_kmeans.segmentor
            self.preprocessor = hfc_kmeans.preprocessor(
                                     model=self.model,
                                     model_config=self.model_config,
                                     out_dir=self.out_dir,
                                     logger=self.logger,
                                     **self.seg_config.hfc_prep_args
                                )

        self.fixed_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

    # -------------------------------------------------------------------------

    def load_trainer(self):
        """
        -----------------------------------------------------------------------

        :return:
        -----------------------------------------------------------------------
        """
        self.logger.info("")
        self.logger.info("*"*20)
        self.logger.info("Loading Trainer ... ")

        # load trainer configuration
        self.trainer_config = load_config(self.configs['trainer'],
                                          'trainer_config')
        self.logger.info("Trainer Specifications:")

        for k in dir(self.trainer_config):
            if k[:2] != "__":
                self.logger.info(f"{k}: {self.trainer_config.__dict__[k]}")
        self.logger.info("*"*20)
    # -------------------------------------------------------------------------

    def setup_trainer(self):
        """
        -----------------------------------------------------------------------
        :return:
        -----------------------------------------------------------------------
        """
        # Optimizer - always chosen as Adam optimizer
        self.optimizer = torch.optim.Adam(self.segmentor.parameters(),
                                          lr=self.trainer_config.lr,
                                          betas=(self.trainer_config.beta1,
                                                 self.trainer_config.beta2)
                                          )

        # Define Losses
        self.seg_loss = self.get_segmentor_loss(self.trainer_config.losses,
                                                self.trainer_config.lambdas)

        # define Scheduler
        self.lr_scheduler = self.get_lr_scheduler(
                                self.trainer_config.scheduler_type,
                                self.trainer_config.scheduler_args)

    # -------------------------------------------------------------------------

    def get_segmentor_loss(self, loss_list, lambdas_list):
        """
        -----------------------------------------------------------------------
        :param loss_list:
        :param lambdas_list:
        :return:
        -----------------------------------------------------------------------
        """

        assert len(loss_list)==len(lambdas_list)

        l_sum = sum(lambdas_list)
        lambdas_list = [l/l_sum for l in lambdas_list]

        losses = [(l, config_mapper.losses[loss])
                  for loss, l in zip(loss_list,
                                     lambdas_list)]

        return losses
    # -------------------------------------------------------------------------

    def get_lr_scheduler(self, sch_name, sch_args):
        """
        -----------------------------------------------------------------------
        :param sch_name:
        :param sch_args:
        :return:
        -----------------------------------------------------------------------
        """
        scheduler = config_mapper.lr_scheduler[sch_name]

        return scheduler(self.optimizer, **sch_args)
    # -------------------------------------------------------------------------

    def start_gui(self):
        """
        -----------------------------------------------------------------------
        :return:
        -----------------------------------------------------------------------
        """
        self.labeling_gui = OneShotLabellerGUI
    # -------------------------------------------------------------------------

    def get_image_from_latent(self,
                              latent,
                              return_features=False):
        """
        -----------------------------------------------------------------------
        :param latent:
        :param noises:
        :param return_features:
        :return:
        -----------------------------------------------------------------------
        """

        with torch.no_grad():
            img, feat = self.model([latent],
                                   truncation=self.model_config.truncation,
                                   truncation_latent=self.mean_latent,
                                   input_is_latent=True,
                                   randomize_noise=False)

        if return_features:
            return img, feat
        else:
            return img
    # -------------------------------------------------------------------------

    def run_pipeline(self,
                     input_latent=None,
                     input_noises=None,
                     blocks_to_run=['setup', 'train', 'test']):
        """
        -----------------------------------------------------------------------

        :return:
        -----------------------------------------------------------------------
        """

        if 'setup' in blocks_to_run:

            # load test samples
            if hasattr(self.model_config, 'sample_noises'):
                self.test_latents = torch.load(self.model_config.sample_latents)
            else:
                self.test_latents, \
                _ = torch.load(self.model_config.sample_latents)

            self.test_labels = torch.load(self.model_config.sample_labels)

            if not isinstance(self.test_labels, torch.Tensor):
                self.test_labels = torch.from_numpy(self.test_labels)

            self.test_indices = range(self.test_labels.shape[0])

            # select / create one sample for training
            ind = self.model_config.one_shot_ind
            self.one_shot_latent = self.test_latents[ind, :]

            if self.test_labels.max()<1:
                self.test_labels = self.test_labels*255

            # special case for LSUN cars
            if self.model_str.find('p-car')!=-1:
                lbl = torch.zeros(self.test_labels.shape[0],
                                  self.test_labels.shape[2],
                                  self.test_labels.shape[2])
                lbl[:,256-192:256+192,:] = self.test_labels.clone()
                self.test_labels = lbl.clone()

            self.one_shot_label = self.test_labels[ind:ind + 1, :, :]

            if self.mode=='online':

                if input_latent is not None:
                    input_noises = input_noises if input_noises is not None \
                                                else self.model.make_noise()

                    self.one_shot_latent = input_latent.clone()
                    self.one_shot_noise  = input_noises.clone()

                if hasattr(self.model_config, 'sample_noises'):
                    self.one_shot_img, self.one_shot_features = \
                        self.model([self.one_shot_latent.unsqueeze(0)],
                                   input_is_latent=True,
                                   randomize_noise=False)

                else:
                    self.one_shot_img, self.one_shot_features  = \
                                self.model([
                                    self.one_shot_latent],
                                    truncation=self.model_config.truncation,
                                    truncation_latent=self.mean_latent,
                                    input_is_latent=True,
                                    randomize_noise=False
                                )

                self.logger.info("Initializing GUI ...")
                self.start_gui()
                self.labeller = self.labeling_gui(self.transform_im_for_gui(
                                                    self.one_shot_img),
                                                  self.model_config.classes)

                self.one_shot_label = self.labeller.get_labels()
                self.one_shot_label = torch.from_numpy(self.one_shot_label).to(
                                      self.one_shot_img.device)

            else:
                if input_latent is not None:
                    raise AttributeError(
                        'Cannot feed input latents in offline mode!')

                if hasattr(self.model_config, 'sample_noises'):
                    self.one_shot_img, self.one_shot_features = \
                        self.model([self.one_shot_latent.unsqueeze(0)],
                                   input_is_latent=True,
                                   randomize_noise=False)
                else:
                    self.one_shot_img, self.one_shot_features = \
                                self.model(
                                    [self.one_shot_latent],
                                    truncation=self.model_config.truncation,
                                    truncation_latent=self.mean_latent,
                                    input_is_latent=True,
                                    randomize_noise=False)

            self.one_shot_noise = [torch.randn(f.shape[0],
                                               1,
                                               f.shape[2],
                                               f.shape[3])
                                   for f in self.one_shot_features]

            if input_latent is None:
                self.test_indices = list(range(self.test_latents.shape[0]))
                self.test_indices.remove(ind)

                self.test_latents = torch.cat([self.test_latents[:ind,:],
                                               self.test_latents[ind+1:]], 0)

                self.test_labels  = torch.cat([self.test_labels[:ind,:,:],
                                               self.test_labels[ind+1:,:,:]])

            if self.num_test_samples is None:
                self.num_test_samples = self.test_labels.shape[0]

        if 'train' in blocks_to_run:
            self.run_trainer()

        if 'test' in blocks_to_run:
            self.run_tests()
    # -------------------------------------------------------------------------

    def run_trainer(self):
        """
        -----------------------------------------------------------------------
        :return:
        -----------------------------------------------------------------------
        """

        if self.train_str=='supervised':

            if self.seg_str in ['repurposegan', 'datasetgan']:
                self.one_shot_features = self.preprocessor(
                                            self.one_shot_features,
                                            self.seg_config.n_layers)

                self.segmentor = self.segmentor(**self.seg_config.seg_args,
                                      n_class=len(self.model_config.classes),
                                      in_ch=self.one_shot_features.shape[1])

            elif self.seg_str=='hfc_kmeans':

                if self.seg_config.train_hfc:
                   self.preprocessor.train_hfc_model(self.one_shot_latent)

                self.one_shot_features, _ = \
                   self.preprocessor.predict_hfc_vectors(self.one_shot_latent)

                self.segmentor = self.segmentor(**self.seg_config.seg_args,
                                     n_class=len(self.model_config.classes))

                self.logger.info(self.segmentor.__str__())

            elif self.seg_str =='hfc_with_simclr' or self.seg_str.find('hfc_with_swav')!=-1:

                if not isinstance(self.preprocessor, hfc_with_swav.preprocessor):
                    self.preprocessor = self.preprocessor(
                                                model=self.model,
                                                model_config=self.model_config,
                                                out_dir=self.out_dir,
                                                logger=self.logger,
                                                tb=self.summary_writer,
                                                **self.seg_config.hfc_prep_args
                                                )

                if self.seg_config.train_hfc:
                    self.preprocessor.preprocess(self.one_shot_latent)

                if self.seg_str.find('hfc_with_swav')!=-1:
                    self.one_shot_features, _ = \
                        self.preprocessor.predict_swav_codes(self.one_shot_latent)
                elif self.seg_str=='hfc_with_simclr':
                    self.one_shot_features, _ = \
                        self.preprocessor.predict_simclr_codes(self.one_shot_latent)

                self.one_shot_features = \
                    self.one_shot_features.detach().clone()

                if isinstance(self.segmentor, hfc_with_swav.segmentor):
                    del self.segmentor
                    torch.cuda.empty_cache()
                    self.segmentor = hfc_with_swav.segmentor(
                                            **self.seg_config.seg_args,
                                            n_class=len(self.model_config.classes)
                    )
                else:
                    self.segmentor = self.segmentor(**self.seg_config.seg_args,
                                         n_class=len(self.model_config.classes))

            self.segmentor.train().to(self.device)
            self.setup_trainer()

            num_samples = self.one_shot_img.shape[0]

            start_time = time.time()

            for epoch in range(self.trainer_config.num_epochs):

                sample_order = list(range(num_samples))
                random.shuffle(sample_order)

                for idx in sample_order:
                    features = self.one_shot_features[
                               idx,:,:,:].unsqueeze(0).to(self.device)

                    if len(self.one_shot_label[idx,:,:].shape)==2:
                        label  = self.one_shot_label[
                                 idx,:,:].unsqueeze(0).to(self.device,
                                                          torch.long)
                    else:
                        label  = self.one_shot_label[idx,:,:].to(self.device,
                                                                 torch.long)

                    c_feat = features.shape[1]

                    out = self.segmentor(features)
                    out = transforms.Resize(self.model_config.image_size)(out)
                    label = transforms.Resize(self.model_config.image_size,
                                              interpolation=Image.NEAREST)(label)

                    loss = 0.

                    for alpha, lf in self.seg_loss:
                        loss = loss + alpha*lf()(out, label)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if epoch % self.trainer_config.print_freq == 0:
                    self.logger.info(f'{epoch:5}-th epoch | '
                                f'loss: {loss.item():6.4f} | '
                                f'time: {time.time() - start_time:6.1f}sec')

                self.lr_scheduler.step()

            self.logger.info('******* Training Complete ********')
            self.logger.info("")
    # -------------------------------------------------------------------------

    def transform_im_for_gui(self, im):
        """
        -----------------------------------------------------------------------
        :param im:
        :return:
        -----------------------------------------------------------------------
        """
        return im.clamp_(-1., 1.).detach().permute(
                                    0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
    # -------------------------------------------------------------------------

    def run_tests(self):
        """
        -----------------------------------------------------------------------
        :return:
        -----------------------------------------------------------------------
        """
        self.test_dir     = os.path.join(self.out_dir, 'tests')
        self.test_img_dir = os.path.join(self.test_dir, 'images')

        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        if not os.path.exists(self.test_img_dir):
            os.mkdir(self.test_img_dir)

        self.segmentor.eval().to('cpu')

        pred_labels = []
        inference_times = []
        for bs in tqdm(range(0, self.num_test_samples, MAX_TEST_BATCH)):
            t0 = time.time()

            test_latents = self.test_latents[bs:bs + MAX_TEST_BATCH, :]
            ims, features = self.model([test_latents],
                                       truncation=self.model_config.truncation,
                                       truncation_latent=self.mean_latent,
                                       input_is_latent=True,
                                       randomize_noise=False)

            if self.seg_str in ['repurposegan', 'datasetgan']:
                features = self.preprocessor(features,
                                         self.seg_config.n_layers).to('cpu')

            elif self.seg_str == 'hfc_kmeans':
                features, _ = self.preprocessor.predict_hfc_vectors(
                                                            test_latents)
                features    = features.to('cpu')
            elif self.seg_str=='hfc_with_simclr' or self.seg_str.find('hfc_with_swav')!=-1:

                if self.seg_str.find('hfc_with_swav')!=-1:
                    features, out_labels = \
                        self.preprocessor.predict_swav_codes(test_latents)
                elif self.seg_str == 'hfc_with_simclr':
                    features, out_labels = \
                        self.preprocessor.predict_simclr_codes(test_latents)

                if len(out_labels.shape)==3:
                    out = transforms.ToPILImage()(
                        out_labels[0, :, :] / out_labels.max())
                else:
                    out = transforms.ToPILImage()(
                        out_labels[0,-1, :, :] / out_labels.max())
                plt.figure()
                plt.subplot(121)
                plt.imshow(out, cmap='jet')
                plt.subplot(122)
                plt.imshow(
                    transforms.ToPILImage()(ims[0, :, :, :] / ims.max()))
                plt.savefig(os.path.join(self.out_dir,
                                         f'test_pred_{bs}.png'))
                plt.close()

                self.summary_writer.add_image('one_shot/test_image',
                                              ims[0, :, :, :] / ims.max())
                self.summary_writer.add_image('one_shot/swav_output',
                                              transforms.ToTensor()(out))

                features = features.to('cpu')

            pred = self.segmentor(features).to('cpu')
            pred = pred.data.max(1)[1]
            pred_labels.append(pred)
            self.summary_writer.add_image('one_shot/predictions',
                                          pred)
            torch.cuda.empty_cache()

            inference_times.append(time.time()-t0)

        pred_labels = torch.cat(pred_labels, axis=0)

        torch.save(pred_labels, os.path.join(self.test_dir,
                                             'label_predictions.pt'))

        ims = [[], [], []]

        results = dict()

        for i in range(self.num_test_samples):
            t0 = time.time()

            input_im = self.get_image_from_latent(
                                self.test_latents[i].unsqueeze(0))
            input_im = transforms.Resize(
                                self.model_config.image_size)(input_im)
            input_im = torch.squeeze(input_im).permute(1, 2, 0).cpu().numpy()

            gt_mask = torch.squeeze(transforms.Resize(
                self.model_config.image_size,
                interpolation=Image.NEAREST)(
                                self.test_labels[i:i+1, :, :])).cpu().numpy()
            pred_mask = transforms.Resize(self.model_config.image_size,
                                          interpolation=Image.NEAREST)(
                                          pred_labels)[i:i+1, :, :].cpu().numpy()[0,:,:]

            disp_im = np.clip(input_im, -1, 1)

            create_pil_collage([np.uint8((disp_im-disp_im.min())
                                         /(disp_im.max()-disp_im.min())*255),
                                np.uint8(
                                    visualize_label_mask(gt_mask,
                                                         self.color_map)*255),
                                np.uint8(
                                    visualize_label_mask(pred_mask,
                                                         self.color_map)*255)],
                                os.path.join(self.test_img_dir,
                                             f'sample_{i}_pred.png'))

            gt_fg_mask = gt_mask.copy()
            gt_fg_mask[gt_fg_mask>0] = 1

            pred_fg_mask = pred_mask.copy()
            pred_fg_mask[pred_fg_mask>0] = 1

            create_pil_collage([np.uint8((disp_im-disp_im.min())
                                         /(disp_im.max()-disp_im.min())*255),
                                np.uint8(
                                    visualize_label_mask(gt_fg_mask,
                                                         self.color_map)*255),
                                np.uint8(
                                    visualize_label_mask(pred_fg_mask,
                                                         self.color_map)*255)],
                                os.path.join(self.test_img_dir,
                                             f'sample_{i}_pred_fg.png'))

            ims[0].append(input_im)
            ims[1].append(gt_mask)
            ims[2].append(pred_mask)

            if self.test_str in ['iou', 'all']:

                mask_iou = {c: get_mask_iou(gt_mask,
                                            pred_mask,
                                            i)
                            for i, c in enumerate(self.model_config.classes)}
                bb_iou =   {c: get_bb_iou(gt_mask,
                                          pred_mask,
                                          k)
                            for k, c in enumerate(self.model_config.classes)}
                w_iou  = get_weighted_iou(gt_mask, mask_iou,
                                          self.model_config.classes)

                w_iou_bin  = get_bin_iou(gt_mask, pred_mask)

                results['bin_iou'] = w_iou_bin

                if i==0: results['mask_iou'] = [mask_iou]
                else:    results['mask_iou'].append(mask_iou)

                if i==0: results['bb_iou'] = [bb_iou]
                else:    results['bb_iou'].append(bb_iou)

                if i==0: results['w_iou'] = [w_iou]
                else:    results['w_iou'].append(w_iou)

            if self.test_str in ['dice', 'all']:
                mask_dice = {c: get_mask_dice(gt_mask,
                                              pred_mask,
                                              i)
                             for i, c in enumerate(self.model_config.classes)}
                bb_dice = {c: get_bb_dice(gt_mask,
                                          pred_mask,
                                          k)
                           for k, c in enumerate(self.model_config.classes)}

                if i == 0:  results['mask_dice'] = [mask_dice]
                else:       results['mask_dice'].append(mask_dice)

                if i == 0:  results['bb_dice'] = [bb_dice]
                else:       results['bb_dice'].append(bb_dice)

        if self.test_str in ['iou', 'all']:
            mask_iou_pd = pd.DataFrame(
                            data=np.array([[s[k]
                                           for k in self.model_config.classes]
                                           for s in results['mask_iou']]),
                                       columns=self.model_config.classes)

            bb_iou_pd = pd.DataFrame(
                            data=np.array([[s[k]
                                           for k in self.model_config.classes]
                                           for s in results['bb_iou']]),
                                     columns=self.model_config.classes)

            mask_iou_pd.to_csv(os.path.join(self.test_dir,
                                            'mask_iou_results.csv'))
            bb_iou_pd.to_csv(os.path.join(self.test_dir,
                                          'bb_iou_results.csv'))

            self.logger.info('\n'
                             + 'Mask IoU Results:\n'
                             + mask_iou_pd.mean(axis=0).__str__())

            self.logger.info('\n'
                             + 'Mean Mask IoU:\n'
                             + mask_iou_pd.mean(axis=0).mean().__str__())

            self.logger.info('\n'
                             + 'Weighted IoU Results:\n'
                             + f'{np.mean(results["w_iou"])}')
            self.logger.info(f'FG IoU: {results["bin_iou"]}')

        self.logger.info(f'Mean Inference Time: {np.mean(inference_times)}')

        if self.test_str in ['iou_vs_pd', 'all']:

            pd_scores = get_pd_at_iou_threshold(
                                    mask_iou_pd,
                                    iou_thr=0.5,
                                    classes=self.model_config.classes)
            results['pd'] = pd_scores

            self.logger.info("Mean PD at IoU=0.5:")

            for k,v in pd_scores.items():
                self.logger.info(f"{k}: \t{v}")

            self.logger.info(f"Mean PD:{np.mean([v for k, v in pd_scores.items()])}")

            iou_vs_pd_curve = get_iou_vs_pd_curve(
                                    iou_pd=mask_iou_pd,
                                    classes=self.model_config.classes)
            results['iou_pd_curve'] = iou_vs_pd_curve

            plot_iou_vs_pd_curve(iou_vs_pd_curve,
                                 self.model_config.classes+['Mean'],
                                 os.path.join(self.test_dir,
                                              'iou_vs_pd_curve.png'),
                                 self.model_str)

        if self.test_str in ['demo']:

            input_im = self.get_image_from_latent(
                                self.one_shot_latent.unsqueeze(0))

            input_im = transforms.Resize(
                                self.model_config.image_size)(input_im)
            input_im = torch.squeeze(input_im).permute(1, 2, 0).cpu().numpy()

            disp_im = np.clip(input_im, -1, 1)

            disp_im_in = np.uint8((disp_im-disp_im.min())
                                  /(disp_im.max()-disp_im.min())*255)

            pred_mask_in = transforms.Resize(
                                  self.model_config.image_size)(self.one_shot_label)

            pred_mask_in = np.uint8(visualize_label_mask(
                                   pred_mask_in.cpu().squeeze(0),
                                   self.color_map)*255)

            for k in range(self.num_test_samples):

                disp_im_1 = ims[0][k]
                disp_im_1 = np.clip(disp_im_1, -1, 1)
                disp_im_1 = np.uint8((disp_im_1
                                      - disp_im_1.min())
                                     / (disp_im_1.max()
                                        - disp_im_1.min()) * 255)

                ims[0][k] = disp_im_1

                pred_im_1 = ims[2][k]
                pred_im_1 = np.uint8(visualize_label_mask(pred_im_1,
                                                          self.color_map) * 255)
                ims[2][k] = pred_im_1

            im_list = [disp_im_in] + ims[0] + [pred_mask_in] + ims[2]

            plt.figure()

            fig = plt.imshow(create_pil_collage(im_list,
                                          os.path.join(self.test_dir,
                                                       'demo.png'),
                                          (2, self.num_test_samples+1),
                                          return_im=True))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.show()

        np.savez_compressed(os.path.join(self.test_dir, 'results.npz'),
                            **results)
    # -------------------------------------------------------------------------


if __name__=="__main__":

    expt_dir = '/mnt/cloudNAS3/Ankit/ganecdotes_expts_2023/'
    out_dir = os.path.join(expt_dir, 'testing_pipeline')
    ename = 'DatasetGAN with n_layers - 5, Size - S, n_epochs - 500'

    one_shot_pipeline = OneShotPipeline(out_dir=out_dir,
                                        exp_name=ename,
                                        segmentor='hfc_with_swav',
                                        tester='demo',
                                        num_test_samples=6,
                                        mode='online'
                                        )
    one_shot_pipeline.run_pipeline()
