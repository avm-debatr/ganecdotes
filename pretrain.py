import argparse
from src.one_shot_pipeline import *
plt.switch_backend('agg')


parser = argparse.ArgumentParser(
    description="Script to pre-train self-supervised clustering model "
                "for one-shot segmentation. User must specify the "
                "StyleGAN model/ds for pre-training "
                "{ffhq-256 | cat-256 | afhq-256 | "
                " horse-256 | car-512 | "
                " pidray-256 | pidray-pliers-256 |"
                " pidray-hammer-256 | pidray-powerbank-256 |"
                " pidray-wrench-256 | pidray-handcuffs-256 } "
                " and "
                "method {hfc_with_swav | hfc_with_simclr | hfc_kmeans}."
                "Training parameters are specified in config files saved as "
                "configs/segmentors/*_config.py."
                "Users can create their own custom config files for their "
                "own models by adding their path in configs/mapper.py"
        )

parser.add_argument("--model",
                    default='ffhq-256',
                    help='StyleGAN model: {ffhq-256 | cat-256 | '
                         'afhq-256 | horse-256 | car-512 |'
                         'pidray-256 | pidray-pliers-256 |'
                         'pidray-hammer-256 | pidray-powerbank-256 |'
                         'pidray-wrench-256 | pidray-handcuffs-256}',
                    choices=['ffhq-256',
                             'cat-256',
                             'afhq-256',
                             'horse-256',
                             'car-512',
                             'pidray-256',
                             'pidray-pliers-256',
                             'pidray-hammer-256',
                             'pidray-powerbank-256',
                             'pidray-wrench-256',
                             'pidray-handcuffs-256'
                             ],
                    type=str
                    )

parser.add_argument("--method",
                    default='hfc_with_swav',
                    help='Method for clustering: '
                         '{hfc_with_swav |'
                         ' hfc_with_simclr |'
                         ' hfc_kmeans}',
                    choices=['hfc_with_swav',
                             'hfc_with_simclr',
                             'hfc_kmeans'],
                    type=str
                    )

parser.add_argument("--out_dir",
                    default="results/pretrain_default_ffhq/",
                    help='Output directory for saving results')

parser.add_argument("--expt_desc",
                    default="Testing Clustering Model",
                    help='Experiment description')

parser.add_argument("--num_test_samples",
                    default=10,
                    help='Number of samples to be tested',
                    type=int)

args = parser.parse_args()

if args.method=='hfc_with_swav' and args.model=='ffhq-256':
    args.method = 'hfc_with_swav_ffhq'
if args.method=='hfc_with_swav' and args.model=='cat-256':
    args.method = 'hfc_with_swav_cat'
if args.method=='hfc_with_swav' and args.model=='car-512':
    args.method = 'hfc_with_swav_car'
if args.method=='hfc_with_swav' and args.model=='horse-256':
    args.method = 'hfc_with_swav_horse'
if args.method=='hfc_with_swav' and args.model.find('pidray')!=-1:
    args.method = 'hfc_with_swav_pidray'

one_shot_pipeline = OneShotPipeline(out_dir=args.out_dir,
                                    exp_name=args.expt_desc,
                                    model=args.model,
                                    segmentor=args.method,
                                    num_test_samples=args.num_test_samples)

one_shot_pipeline.seg_config.train_hfc = True
one_shot_pipeline.seg_config.hfc_prep_args['train'] = True

if args.method=='hfc_kmeans':
    one_shot_pipeline.seg_config.hfc_prep_args['presaved'] = False

one_shot_pipeline.run_pipeline()
