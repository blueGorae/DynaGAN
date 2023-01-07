
import os
import glob
import json
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from pathlib import Path
from models.DynaGAN import DynaGAN
from utils.file_utils import save_images
from options.DynaGAN_options import DynaGANOptions
toPIL = torchvision.transforms.ToPILImage()
import random

def make_label(batch, c_dim, device, label = None):
    c = torch.zeros(batch, c_dim).to(device)
    
    if label is not None:
        c_indicies = [label for _ in range(batch)]
    else:
        c_indicies = torch.randint(0, c_dim, (batch,))
        
    for i, c_idx in enumerate(c_indicies):
        c[i,c_idx] = 1.0
    
    return c


def train(args, output_dir):
    # Set up networks, optimizers.
    print("Initializing networks...")
    net = DynaGAN(args)
    style_latent = net.embed_style_img(args.style_img_dir)
    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr = args.lr,
        betas = (0, 0.99)
    )
    sample_dir = os.path.join(output_dir, "sample")
    ckpt_dir = os.path.join(output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Set fixed sample
    fixed_z = torch.randn(args.n_sample, 512, device=args.device)
    np.save("test_latent.npy",fixed_z.cpu().numpy())
    save_image(net, fixed_z, args, sample_dir, -1)

    # Training loop
    pbar = tqdm(range(args.iter))
    for i in pbar:

        net.train()

        sample_z = torch.randn(args.batch, 512, device=args.device)

        sample_domain_label = make_label(args.batch, c_dim=args.c_dim, device=args.device)

        
        if args.use_truncation_in_training:
            [_, _, _, _], loss = net([sample_z], truncation=args.sample_truncation, domain_labels=[sample_domain_label])
        else:
            [_, _, _, _], loss = net([sample_z], domain_labels=[sample_domain_label])

        net.zero_grad()
        loss.backward()
        g_optim.step()
        
        pbar.set_description(f"Finetuning Generator | Total loss: {loss}")

        if ((i + 1) % args.vis_interval == 0 or (i + 1) == args.iter):
            save_image(net, fixed_z, args, sample_dir, i)

        if args.save_interval is not None and ((i + 1) % args.save_interval == 0 or (i + 1) == args.iter):
            ckpt_name = '{}/{}.pt'.format(ckpt_dir, str(i + 1).zfill(6))
            save_checkpoint(net, g_optim, style_latent, ckpt_name)

    ckpt_name = '{}/{}.pt'.format(ckpt_dir, "final")
    save_checkpoint(net, g_optim, style_latent, ckpt_name)




def save_image(net, fixed_z, args, sample_dir, i):
    net.eval()
    with torch.no_grad():
        for domain_idx in range(args.c_dim):
            domain_label = make_label(args.n_sample, c_dim=args.c_dim, device=args.device, label=domain_idx)
            [sampled_src, sampled_dst, rec_dst, without_color_dst], loss = net([fixed_z],
                                                                        truncation=args.sample_truncation,
                                                                        domain_labels=[domain_label],
                                                                        inference=True)
            grid_rows = int(args.n_sample ** 0.5)
            save_images(sampled_dst, sample_dir, f"dst_{domain_idx}", grid_rows, i+1)
            save_images(without_color_dst, sample_dir, f"without_color_{domain_idx}", grid_rows, i+1)
            save_images(rec_dst, sample_dir, f"rec_{domain_idx}", grid_rows, i+1)
        
        
def save_checkpoint(net, g_optim, style_latent, ckpt_name, is_dynagan=True):
    save_dict = {
            "g_ema": net.generator_trainable.generator.state_dict(),
            "g_optim": g_optim.state_dict(),
            "latent_avg": net.generator_trainable.mean_latent,
            "style_latent": style_latent,
            "is_dynagan": is_dynagan,
            "c_dim": net.generator_trainable.c_dim,
        }
    
    torch.save(
        save_dict, 
        ckpt_name
    )


if __name__ == "__main__":

    option = DynaGANOptions()
    parser = option.parser

    # I/O arguments
    parser.add_argument('--style_img_dir', type=str, default="target_data/raw_data",
                        help='Style image')
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument("--lambda_contrast", type=float, default=1.0, help="Weight of contrastive loss")
    parser.add_argument("--lambda_id", type=float, default=3.0, help="Weight of idtentity loss")
    parser.add_argument("--load_inverted_latents",  action='store_true', help="load inverted latents")
    parser.add_argument("--no_scaling",  action='store_true', help="no filter scaling")
    parser.add_argument("--no_residual",  action='store_true', help="no residual scaling")
    parser.add_argument("--id_model_path",  type=str, default="pretrained_models/model_ir_se50.pth", help="identity path")
    parser.add_argument("--human_face", action='store_true', help="Whether it is for human faces")
    args = option.parse()
    
    args.style_img_dir = glob.glob(os.path.join(args.style_img_dir, "*.jpg")) + glob.glob(os.path.join(args.style_img_dir, "*.png")) + glob.glob(os.path.join(args.style_img_dir, "*.jpeg"))
    args.c_dim = len(args.style_img_dir)
    print(f"Number of domains: {args.c_dim}")
    output_dir = args.output_dir # os.path.join(args.output_dir, Path(args.style_img).stem)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    import time

    start_time = time.time()
    train(args, output_dir)
    end_time = time.time()
    print(f"Training time {end_time-start_time}s")
