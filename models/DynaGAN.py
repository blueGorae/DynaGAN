import sys
# sys.path.append('/root/workspace/nips2022/main/MindTheGap')
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from models.stylegan2.model import Generator
from losses.clip_loss import CLIPLoss
from losses.id_loss import IDLoss
from losses import lpips
import PIL
import torchvision
from utils.bicubic import BicubicDownSample
from utils.model_utils import google_drive_paths, download_weight
import dlib
from utils.shape_predictor import align_face
from models.II2S import II2S
from pathlib import Path
from datasets.target_dataset import TargetDataset
toPIL = torchvision.transforms.ToPILImage()
import time

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class SG2Generator(torch.nn.Module):
    def __init__(self, checkpoint_path, latent_size=512, map_layers=8, img_size=256, channel_multiplier=2, device='cuda:0', is_dynagan=False, c_dim=0, no_scaling=False, no_residual=False):
        super(SG2Generator, self).__init__()

        self.latent_size = latent_size
        self.map_layers  = map_layers
        self.img_size = img_size
        self.device = device
        self.c_dim = c_dim
        
        

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.mean_latent = checkpoint['latent_avg']

        
        self.no_scaling = no_scaling
        self.no_residual = no_residual
        
        if "is_dynagan" not in checkpoint.keys():
            self.generator = Generator(
                img_size, latent_size, map_layers, channel_multiplier=channel_multiplier, multi_domain=is_dynagan, c_dim=c_dim
            ).to(device)
            self.generator.load_state_dict(checkpoint["g_ema"], strict=True)
            if is_dynagan:
                self.generator.create_domain_modulation(no_scaling=no_scaling, no_residual=no_residual)
        else:
            self.generator = Generator(
                img_size, latent_size, map_layers, channel_multiplier=channel_multiplier, multi_domain=is_dynagan, c_dim=c_dim
            ).to(device)
            self.generator.create_domain_modulation(no_scaling=no_scaling, no_residual=no_residual)
            self.generator.load_state_dict(checkpoint["g_ema"], strict=True)
            
        self.is_dynagan = True

    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):

        if self.is_dynagan:
            layers = [conv.conv.hypernet for conv in self.generator.convs] + [self.generator.conv1.conv.hypernet] + [self.generator.embedding]
            return layers
        
        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])
        if phase == 'shape':
            # layers 1-2
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        if phase == 'no_fine':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        if phase == 'shape_expanded':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers()
        else:
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def style(self, styles):
        '''
        Convert z codes to w codes.
        '''
        styles = [self.generator.style(s) for s in styles]
        return styles

    def embedding(self, domain_labels):
        '''
        Convert z codes to w codes.
        '''
        domain_styles = [self.generator.embedding(l) for l in domain_labels]
        return domain_styles

    def modulation_layers(self):
        return self.generator.modulation_layers

    def forward(self,
        styles,
        return_latents=False,
        truncation=1,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        domain_labels = [None],
        alpha=1.0, beta=1.0,
        domain_is_latents=False,
        ):
        return self.generator(styles, return_latents=return_latents, truncation=truncation,
                              truncation_latent=self.mean_latent, noise=noise, randomize_noise=randomize_noise,
                              input_is_latent=input_is_latent, domain_labels=domain_labels, alpha=alpha, beta=beta, domain_is_latents=domain_is_latents)

class DynaGAN(torch.nn.Module):
    def __init__(self, args):
        super(DynaGAN, self).__init__()
        self.args = args

        self.device = args.device

        # Set up frozen (source) generator
        if not os.path.exists(args.frozen_gen_ckpt):
            download_weight(args.frozen_gen_ckpt)
        self.generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size).to(self.device)
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()

        # Set up trainable (target) generator
        self.generator_trainable = SG2Generator(args.train_gen_ckpt, img_size=args.size, c_dim=args.c_dim, no_scaling=args.no_scaling, no_residual=args.no_residual, is_dynagan=True).to(self.device)
        self.generator_trainable.freeze_layers()
        self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))

        self.generator_trainable.train()
        self.generator_trainable.mean_latent = self.generator_frozen.mean_latent

        # Set up losses
        self.clip_loss_models = {model_name: CLIPLoss(self.device,
                                                      lambda_direction=args.lambda_direction,
                                                      lambda_patch=args.lambda_patch,
                                                      lambda_global=args.lambda_global,
                                                      lambda_manifold=args.lambda_manifold,
                                                      lambda_texture=args.lambda_texture,
                                                      lambda_contrast=args.lambda_contrast,
                                                      clip_model=model_name)
                                 for model_name in args.clip_models}

        self.clip_model_weights = {model_name: weight for model_name, weight in
                                   zip(args.clip_models, args.clip_model_weights)}
        self.mse_loss = torch.nn.MSELoss()

        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
        self.D_VGG = BicubicDownSample(factor=int(args.size/256))
        self.mse = torch.nn.MSELoss()
        
        if self.args.lambda_id >0.:
            self.id_loss = IDLoss(self.args.id_model_path)
        
        self.n_latent = self.generator_frozen.generator.n_latent

    def embed_style_img(self, style_img_dir):
        if self.args.human_face:
            from options.face_embed_options import II2S_s_opts
        else:
            from options.cat_embed_options import II2S_s_opts

        ZP_input_imgs = []
        ZP_input_imgs_256 = []
        ZP_imgs_tensor = []
        ZP_imgs_tensor_256 = []
        
        
        
        for style_img in style_img_dir:
            # aligned_im_path = os.path.join('style_images', 'Aligned', Path(style_img).stem + '.png')
            # unaligned_im_path = os.path.join('style_images', 'Unaligned', style_img)
            # if os.path.exists(aligned_im_path) or os.path.exists(unaligned_im_path):
            #     if not os.path.exists(aligned_im_path):
            #         predictor_weight = os.path.join('pretrained_models', 'shape_predictor_68_face_landmarks.dat')
            #         download_weight(predictor_weight)
            #         predictor = dlib.shape_predictor(predictor_weight)

            #         face = align_face(unaligned_im_path, predictor, output_size=II2S_s_opts.output_size)
            #         face.save(aligned_im_path)

                #########################
            ZP_input_img = PIL.Image.open(style_img).convert('RGB')
            ZP_input_imgs.append(ZP_input_img)
            
            ZP_input_img_256 = ZP_input_img.resize((256, 256), PIL.Image.LANCZOS)
            ZP_input_imgs_256.append(ZP_input_img_256)
            
            ZP_img_tensor = 2.0 * torchvision.transforms.ToTensor()(ZP_input_img).unsqueeze(0).cuda() - 1.0
            ZP_imgs_tensor.append(ZP_img_tensor)
            
            ZP_img_tensor_256 = 2.0 * torchvision.transforms.ToTensor()(ZP_input_img_256).unsqueeze(0).cuda() - 1.0
            ZP_imgs_tensor_256.append(ZP_img_tensor_256)
                
            # else:
            #     sys.exit('Image {} does not exist'.format(style_img))
            
        self.ZP_img_tensor = torch.cat(ZP_imgs_tensor).detach().cpu()
        self.ZP_img_tensor_256 = torch.cat(ZP_imgs_tensor_256).detach().cpu()
            
        if not os.path.exists(II2S_s_opts.ckpt):
            download_weight(II2S_s_opts.ckpt)
        
        if self.args.load_inverted_latents:
            latents = torch.from_numpy(np.load(os.path.join(self.args.output_dir, "inverted_latents.npy"))).type(torch.FloatTensor)
        else:
            ii2s = II2S(II2S_s_opts)
            start_inversion_time = time.time()
            latents = ii2s.invert_images(image_path=style_img_dir, output_dir=None,
                                        return_latents=True, align_input=False, save_output=False)[0]
            end_inversion_time = time.time()
            print(f"inversion time: {end_inversion_time - start_inversion_time}s")
            latents = latents.detach().clone().cpu()
            np.save(os.path.join(self.args.output_dir, "inverted_latents.npy"), latents.cpu().numpy())
            ii2s = ii2s.cpu()
            del ii2s
            torch.cuda.empty_cache()
            
        self.ZP_target_latent = latents
        self.dataset = TargetDataset(ZP_target_latent=latents, ZP_img_tensor=self.ZP_img_tensor, ZP_img_tensor_256=self.ZP_img_tensor_256)

        return self.ZP_target_latent

    def forward(
            self,
            styles,
            truncation=1,
            domain_labels = [None],
            randomize_noise=True,
            inference=False
    ):

        # if self.training:
        #     self.generator_trainable.unfreeze_layers()
        with torch.no_grad():
            w_styles = self.generator_frozen.style(styles)
            frozen_img = self.generator_frozen(w_styles, input_is_latent=True, truncation=1,
                                               randomize_noise=randomize_noise)[0]
            domain_idx = torch.argmax(domain_labels[0], dim=1).tolist()
            ZP_target_latent, new_img, ZP_img_tensor_256 = self.dataset[domain_idx]
            ZP_target_latent, new_img, ZP_img_tensor_256 = ZP_target_latent.cuda(), new_img.cuda(), ZP_img_tensor_256.cuda()
            old_img = self.generator_frozen([ZP_target_latent], input_is_latent=True, truncation=1,
                                        randomize_noise=randomize_noise)[0]
        if inference:
            with torch.no_grad():
                rec_img = self.generator_trainable([self.ZP_target_latent.cuda()], input_is_latent=True, truncation=1, domain_labels = [domain_labels[0][0:1].repeat(len(self.ZP_target_latent), 1)],
                                                   randomize_noise=randomize_noise)[0]
                tmp_latents = w_styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
                without_color_img = self.generator_trainable([tmp_latents], input_is_latent=True, truncation=truncation, domain_labels = domain_labels,
                                                             randomize_noise=randomize_noise)[0]

                tmp_latents = truncation * (
                            w_styles[0] - self.generator_trainable.mean_latent) + self.generator_trainable.mean_latent
                tmp_latents = tmp_latents.unsqueeze(1).repeat(1, self.n_latent, 1)
                
                if not None in domain_labels:
                    domain_idx = torch.argmax(domain_labels[0]).item()
                    tmp_latents[:, 7:, :] = self.ZP_target_latent[domain_idx:domain_idx+1][:, 7:, :].cuda()
                else:
                    tmp_latents[:, 7:, :] = self.ZP_target_latent[:, 7:, :].cuda()
                color_img = self.generator_trainable([tmp_latents], input_is_latent=True, truncation=1, domain_labels = domain_labels,
                                                     randomize_noise=randomize_noise)[0]
                return [frozen_img, color_img, rec_img, without_color_img], None

        else:
            
            rec_img = self.generator_trainable([ZP_target_latent], input_is_latent=True, truncation=1, domain_labels = domain_labels,
                                                randomize_noise=randomize_noise)[0]

            trainable_img = self.generator_trainable(w_styles, input_is_latent=True, truncation=1, domain_labels = domain_labels,
                                                     randomize_noise=randomize_noise)[0]
            clip_across_loss = torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models[
                model_name](frozen_img, old_img, trainable_img, new_img, True) for model_name in
                                                      self.clip_model_weights.keys()]))

            clip_within_loss = torch.sum(
                torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models[model_name](
                    frozen_img, old_img, trainable_img, new_img, False) for model_name in
                             self.clip_model_weights.keys()]))

            ref_clip_loss = torch.sum(torch.stack(
                [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].rec_loss(rec_img, new_img) for
                 model_name in self.clip_model_weights.keys()]))

            l2_loss = self.mse(self.D_VGG(rec_img), ZP_img_tensor_256)
            lpips_loss = self.percept(self.D_VGG(rec_img), ZP_img_tensor_256).mean()

            
            if self.args.lambda_contrast > 0.:
                contrastive_loss = torch.sum(torch.stack(
                    [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].contrastive_adaptation_loss(trainable_img, self.ZP_img_tensor_256.cuda(), domain_labels) for
                    model_name in self.clip_model_weights.keys()]))
            else:
                contrastive_loss = 0.
            
            if self.args.lambda_id > 0.:
                id_loss = self.id_loss(frozen_img, trainable_img)[0]
            else:
                id_loss = 0.
                
            loss = self.args.clip_across_lambda * clip_across_loss + self.args.ref_clip_lambda * ref_clip_loss + \
                   self.args.lpips_lambda * lpips_loss + self.args.l2_lambda * l2_loss + self.args.clip_within_lambda * clip_within_loss + \
                       self.args.lambda_contrast * contrastive_loss + self.args.lambda_id * id_loss

            return [None, None, None, None], loss

