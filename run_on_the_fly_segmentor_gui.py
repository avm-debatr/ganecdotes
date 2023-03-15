from src.one_shot_pipeline import *
from lib.gui.interactive_labeller import *

parser = argparse.ArgumentParser(
    description="Script to run an interactive GUI  "
                "for on-the-fly one-shot segmentation. "
                "User must specify the "
                "StyleGAN model/ds for running the gui"
                "{ffhq-256 | cat-256 | afhq-256 | "
                " horse-256 | car-512 | "
                " pidray-256 | pidray-pliers-256 |"
                " pidray-hammer-256 | pidray-powerbank-256 |"
                " pidray-wrench-256 | pidray-handcuffs-256 }."
                "Users can create their own custom config files for their "
                "own models by adding their path in configs/mapper.py"
                "The GUI contains labelling tools developed using "
                "https://github.com/bryandlee/repurpose-gan."
                "The GUI allows labelling StyleGAN images and "
                "synthesize new annotated images on-the-fly."
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

parser.add_argument("--out_dir",
                    default="data/gui_demo/",
                    help='Expt. directory with saved model + for storing output')

parser.add_argument("--expt_desc",
                    default="Interactive GUI for On-the-fly Segmentation",
                    help='Experiment description')

args = parser.parse_args()

if args.model=='ffhq-256':
    args.method = 'hfc_with_swav_ffhq'
if args.model=='cat-256':
    args.method = 'hfc_with_swav_cat'
if args.model=='car-512':
    args.method = 'hfc_with_swav_car'
if args.model=='horse-256':
    args.method = 'hfc_with_swav_horse'

one_shot_pipeline = OneShotPipeline(out_dir=args.out_dir,
                                    exp_name=args.expt_desc,
                                    model=args.model,
                                    segmentor='hfc_with_swav',
                                    num_test_samples=8)

one_shot_pipeline.seg_config.train_hfc = False
one_shot_pipeline.seg_config.hfc_prep_args['train'] = False

# Change the number of epochs for fine-tuned output - though this will slow down
# the labelling process
one_shot_pipeline.trainer_config.num_epochs = 100

one_shot_pipeline.run_pipeline(blocks_to_run=['setup'])

labeller = InteractiveLabellerGUI(one_shot_learner=one_shot_pipeline,
                                  cmap='jet')
