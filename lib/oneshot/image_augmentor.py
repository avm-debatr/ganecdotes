from lib.util.util import *
from lib.util.visualization import create_pil_collage
from configs import mapper as config_mapper
from models.stylegan2.model import Generator
from torchvision.transforms import ToPILImage


def create_perturbed_vectors_from_latents(input_latents,
                                          model,
                                          n_samples=10,
                                          n_layers=6,
                                          perturb_std=[0.25]*6):
    """
    ---------------------------------------------------------------------------
    Perturb the latent vectors to create new augmented image samples.

    The vectors are perturbed as per styleGAN blocks n

    Total number of latents is 2*n + 1

    0th latent is for initial convolution
    (2*n)th and (2*n+1)th latent for nth StyleGAN block

    number of layers is calculated per style block
    so perturbation is for 2 latents at a time

    perturb_std decides the extent of perturbation for each latents
    - to disable perturbation for a ith latent, set perturb_set[i] = 0

    :param input_latents:   input latent vectors
                            (must be a list of n extended latent vectors,
                             n being the number of conv)
    :param n_samples:
    :param n_layers:
    :param perturb_std:
    :return:
    ---------------------------------------------------------------------------
    """

    perturbed_latents = []

    for n in range(2*n_layers):

        curr_latent = input_latents[0, n, :].clone()
        out_latents = curr_latent.repeat(n_samples, 1)

        noises = model.style(torch.randn_like(out_latents))
        # perturbations = out_latents
        #                 + perturb_std[n]*torch.randn_like(out_latents)
        perturbations =    (1 - perturb_std[n]) * out_latents \
                         + perturb_std[n]       * noises

        perturbed_latents.append(perturbations)

    return perturbed_latents
# -----------------------------------------------------------------------------


def create_images_and_features_from_perturbed_latents(perturbations,
                                                      model,
                                                      model_args,
                                                      layer_no=None,
                                                      return_image=True,
                                                      return_feat=True,
                                                      skip_const=False):
    """
    ---------------------------------------------------------------------------
    :param perturbations:
    :param model:
    :param model_args:
    :return:
    ---------------------------------------------------------------------------
    """

    perturbed_imgs, perturbed_features = model([perturbations],
                                               truncation=model_args['truncation'],
                                               truncation_latent=model_args['mean_latent'],
                                               input_is_latent=True,
                                               randomize_noise=False)
    n_layers = len(perturbed_features)//2

    if skip_const:
        perturbed_features =  [torch.cat([perturbed_features[2*n+1],
                                         perturbed_features[2*n+2]], 1)
                              for n in range(n_layers)]
    else:
        perturbed_features = [perturbed_features[0]] \
                             + [torch.cat([perturbed_features[2*n+1],
                                           perturbed_features[2*n+2]], 1)
                                for n in range(n_layers)]

    # if skip_const and layer_no is not None: layer_no +=1

    if return_feat and return_image:
        return perturbed_imgs, \
               perturbed_features if layer_no is None \
                                  else perturbed_features[layer_no]

    elif return_image and not return_feat:
        return perturbed_imgs

    elif return_feat and not return_image:
        return perturbed_features if layer_no is None \
                                  else perturbed_features[layer_no]

# -----------------------------------------------------------------------------


if __name__=="__main__":

    device       = 'cuda:0'
    model_config = load_config(config_mapper.models['ffhq-256'],
                               'model_config')
    model        = Generator(**model_config.gen_args)

    g_state_dict = torch.load(model_config.model_path, map_location=device)

    if 'g_ema' in g_state_dict.keys():
        model.load_state_dict(g_state_dict["g_ema"], strict=False)
    else:
        model.load_state_dict(g_state_dict, strict=False)

    model.eval()
    model = model.cuda()
    mean_latent = model.mean_latent(model_config.num_latents_for_mean)
    truncation = 0.7

    test_latent = torch.load(model_config.sample_latents)[0]
    # test_latent = model.style(test_latent)
    # test_latent = mean_latent + truncation*(test_latent-mean_latent)
    # test_latent = test_latent.unsqueeze(1).repeat(1, model.n_latent, 1)

    with torch.no_grad():

        k = 5
        test_latent = test_latent[k:k+1,:]

        out_dir = os.path.join(RESULTS_DIR, f'support_set_example_{k}')
        os.makedirs(out_dir, exist_ok=True)

        # orig_img = create_images_from_perturbed_latents(test_latent,
        #                                                 model,
        #                                                 dict(truncation=1,
        #                                                      mean_latent=mean_latent)
        #                                                )

        orig_img, w_latents = model([test_latent],
                                    return_latents=True,
                                    truncation_latent=mean_latent,
                                    truncation=truncation,
                                    input_is_latent=True)
        test_w_latents = w_latents.clone()

        orig_img = ToPILImage()(0.5*(orig_img+1).squeeze())
        orig_img.save(os.path.join(out_dir, 'original_img.png'))

        n_layers = 14
        n_samples = 4
        perturbed_latents = create_perturbed_vectors_from_latents(test_w_latents,
                                                                  model,
                                                                  n_samples=n_samples,
                                                                  n_layers=n_layers,
                                                                  perturb_std=[5, 5]+ [3, 3] + [2]*10)

        for n in range(n_layers):

            new_latents = test_w_latents.repeat(n_samples, 1, 1)
            new_latents[:, n, :] = perturbed_latents[n]

            perturbed_imgs = create_images_and_features_from_perturbed_latents(new_latents,
                                                                     model,
                                                                     {'truncation': 0.7,
                                                                      'mean_latent': mean_latent})

            perturbed_imgs = np.uint8(((perturbed_imgs+1)*0.5*255).permute(0, 2, 3, 1).cpu().numpy())

            create_pil_collage([perturbed_imgs[n,:,:,:] for n in range(perturbed_imgs.shape[0])],
                               os.path.join(out_dir, f'support_set_layer_{n}.png'))

