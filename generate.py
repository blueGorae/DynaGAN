import torch
import numpy as np
import os
import random
from tqdm import tqdm

from models.DynaGAN import SG2Generator
import torchvision
from torchvision.utils import save_image
from argparse import ArgumentParser
toPIL = torchvision.transforms.ToPILImage()



def make_label(batch, c_dim, device, label = None):
    c = torch.zeros(batch, c_dim).to(device)
    
    if label is not None:
        c_indicies = [label for _ in range(batch)]
    else:
        c_indicies = torch.randint(0, c_dim, (batch,))
        
    for i, c_idx in enumerate(c_indicies):
        c[i,c_idx] = 1.0
    
    return c


def main(args):

    # Load finetuned generator
    print('Load finetuned generator')
    
    
    target_ckpt = torch.load(args.ckpt, map_location=args.device)

    style_latent = target_ckpt["style_latent"]
    latent_avg = target_ckpt["latent_avg"].type(torch.FloatTensor).to(device)
    c_dim = target_ckpt['c_dim']
    is_dynagan = target_ckpt['is_dynagan']
    
    generator = SG2Generator(args.ckpt, img_size=args.size, c_dim=c_dim, no_scaling=args.no_scaling, no_residual=args.no_residual, is_dynagan=is_dynagan).to(args.device)
    generator.eval()
    n_latents =  generator.generator.n_latent

    if args.latent_path is None:
        random_z = torch.randn(args.n_sample, 512).to(args.device)
    else:
        random_z = torch.from_numpy(np.load(args.latent_path)).type(torch.FloatTensor).to(args.device)

    with torch.no_grad():
        w_styles = generator.style([random_z])[0]

    output_latents = args.truncation * (w_styles - latent_avg) + latent_avg
    output_latents = output_latents.unsqueeze(1).repeat(1, n_latents, 1)

    # Save generated images
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "source"), exist_ok=True)
    with torch.no_grad():
        outputs = generator([output_latents], input_is_latent=True, randomize_noise=False)[0]
        
        for j in tqdm(range(len(outputs))):
            save_image(
                outputs[j],
                os.path.join(output_dir,"source", f"{str(j).zfill(6)}.png"),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            
        outputs = generator([style_latent], input_is_latent=True, randomize_noise=False)[0]
        for j in tqdm(range(len(outputs))):
            save_image(
                outputs[j],
                os.path.join(output_dir,"source", f"rec_{str(j).zfill(6)}.png"),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
        
    
    with torch.no_grad():
        w_styles = generator.style([random_z])[0]

    output_latents = args.truncation * (w_styles - latent_avg) + latent_avg
    output_latents = output_latents.unsqueeze(1).repeat(1, n_latents, 1)

    # Save generated images
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "target"), exist_ok=True)
    
    with torch.no_grad():
        for i in tqdm(range(c_dim)):

            mixed_latent = output_latents.clone()
            mixed_latent[:, 7:, :] = style_latent[i:i+1][:, 7:, :]
            w = [mixed_latent]

            domain_label = make_label(1, c_dim=c_dim, device=args.device, label=i)
            outputs = generator(w, input_is_latent=True, randomize_noise=False, domain_labels=[domain_label])[0]
            
            for j in range(len(outputs)):
                save_image(
                    outputs[j],
                    os.path.join(output_dir, "target", f"style_{i}_{str(j).zfill(6)}.png"),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
    

    # Save generated images
    os.makedirs(os.path.join(output_dir, "target_wo_style"), exist_ok=True)
    
    with torch.no_grad():
        for i in tqdm(range(c_dim)):
            mixed_latent = output_latents.clone()
            w = [mixed_latent]

            domain_label = make_label(1, c_dim=c_dim, device=args.device, label=i)
            outputs = generator(w, input_is_latent=True, randomize_noise=False, domain_labels=[domain_label])[0]
            
            for j in range(len(outputs)):
                save_image(
                    outputs[j],
                    os.path.join(output_dir, "target_wo_style", f"style_{i}_{str(j).zfill(6)}.png"),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
    
    






if __name__ == '__main__':
    device = 'cuda'

    parser = ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--n_sample', type=int, default=25, help='number of fake images to be sampled')
    parser.add_argument('--n_steps', type=int, default=40, help="determines the granualarity of interpolation")
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="output/checkpoint/final.pt")
    parser.add_argument('--mode', type=str, default='viz_imgs')
    parser.add_argument('--latent_path', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default="samples")
    parser.add_argument("--no_scaling",  action='store_true', help="no filter scaling")
    parser.add_argument("--no_residual",  action='store_true', help="no residual scaling")
    parser.add_argument('--each', action='store_true', default=False)

    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    args.device = "cuda"
    main(args)


