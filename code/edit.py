import os

from network import *
import torch
from argparse import ArgumentParser
import warnings
from functools import partial
from oal_utils import mse_loss2

warnings.filterwarnings("ignore")


def load_and_process_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path).convert('L')
    image = image.resize(target_size, Image.ANTIALIAS)
    image_np = np.array(image)
    image_np = (image_np > 127).astype(np.float32)
    image_tensor = torch.from_numpy(image_np)
    row_indices, col_indices = torch.where(image_tensor == 1)
    coordinates = torch.stack((row_indices, col_indices), dim=1)

    return coordinates

def parse_loss_function(loss_str):
    if loss_str.startswith("mse_loss2"):
        params = loss_str.split(",")
        ls = 1.0
        for param in params[1:]:
            if param.startswith("ls="):
                ls = float(param.split("=")[1])
        return partial(mse_loss2, ls=ls)
    else:
        raise ValueError(f"Unsupported loss function: {loss_str}")

def parse_dict_string(dict_str):
    result = {}
    if not dict_str:
        return result

    if ',' not in dict_str:
        key, value = dict_str.split(":", 1)
        result[int(key)] = value.strip()
        return result

    for item in dict_str.split(","):
        key, value = item.split(":", 1)
        result[int(key)] = value.strip()
    return result


def run_and_displayloss_MDE(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None,
                           verbose=True, i=0, name=None, max_iter=15, scale=2.5, optimization_step=20, oal_lr=1e-1, use_32_blocks=False,
                           ediffi_coeff=0, loss_func=None, idx2words=None,oal_mask=None):
    images, x_t = text2image_ldm_stable_MDE(ldm_stable, prompts, controller, latent=latent,
                                           num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                           generator=generator, uncond_embeddings=uncond_embeddings, max_iter=max_iter,
                                           scale=scale,optimization_step=optimization_step,
                                           oal_lr=oal_lr, use_32_blocks=use_32_blocks,
                                           ediffi_coeff=ediffi_coeff,
                                           loss_func=loss_func, idx2words=idx2words, oal_mask=oal_mask)
    utils.save_image(images[0], "{}/{}.png".format(name, prompts[0]), nrow=1)
    utils.save_image(images[1], "{}/{}.png".format(name, prompts[1]), nrow=1)
    return images, x_t


null_inversion = NullInversion(ldm_stable)

parser = ArgumentParser()

parser.add_argument("--source_prompt", default="a cat wearing a straw hat",type=str)
parser.add_argument("--target_prompt", default="a dog wearing a red hat", type=str)
parser.add_argument("--img_path", default="example/1/cat_hat.jpg", type=str)
parser.add_argument("--oal_mask", default="example/1/hat.png",  nargs="+", type=str)  # for objects change
parser.add_argument("--idx2words", type=str, default="1:dog") # for objects change
parser.add_argument("--ccl_mask", default="example/1/hat.png", nargs="+", type=str)  # for color change
parser.add_argument("--ccl_word", default="red", nargs="+", type=str)  # for color change
parser.add_argument("--negative_prompt", default=None, type=str)
parser.add_argument("--negative_word", default=None, nargs="+", type=str)
parser.add_argument("--result_dir", default="result", type=str)
parser.add_argument("--max_iteration", default=15, type=int)
parser.add_argument("--scale", default=1.5, type=float)
parser.add_argument("--optimization_step", default=20, type=int)
parser.add_argument("--oal_lr", default=1e-1, type=float)  # zestguide arg
parser.add_argument("--use_32_blocks", default=False, type=bool)
parser.add_argument("--ediffi_coeff", default=0, type=float)
parser.add_argument("--loss_func", default="mse_loss2,ls=0.1", type=str)

# end
args = parser.parse_args()

if __name__ == '__main__':
    # null-text inversion
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(args.img_path, args.source_prompt,
                                                                          offsets=(0, 0, 0, 0), verbose=True)

    prompts = [args.source_prompt, args.target_prompt]

    cross_replace_steps = {'default_': .8, }
    self_replace_steps = 0.5

    positive_word = [[args.ccl_mask for p in args.ccl_word], [p for p in args.ccl_word]]
    print("positive_word: ", positive_word)
    negative_word = None
    print("Prompt for Inversion:", args.source_prompt)
    print("Prompt for Editing:", args.target_prompt)

    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, None, name=args.result_dir,
                                 word=positive_word, adj_mask=args.mask_path, word_ng=negative_word)

    os.makedirs(args.result_dir, exist_ok=True)

    loss_func = parse_loss_function(args.loss_func)  # for oal
    idx2words = parse_dict_string(args.idx2words)  # for oal
    mask_path_1 = args.oal_mask[0]
    coords_1 = load_and_process_image(mask_path_1)
    mask = np.zeros((512, 512))
    mask[coords_1[:, 0], coords_1[:, 1]] = 1
    oal_mask = torch.from_numpy(mask)
    images, _ = run_and_displayloss_MDE(prompts, controller, run_baseline=False, latent=x_t,
                                       uncond_embeddings=uncond_embeddings, name=args.result_dir,
                                       max_iter=args.max_iteration, scale=args.scale, optimization_step=args.optimization_step,
                                       oal_lr=args.oal_lr, use_32_blocks=args.use_32_blocks,
                                       ediffi_coeff=args.ediffi_coeff,
                                       loss_func=loss_func, idx2words=idx2words, oal_mask=oal_mask.int())
