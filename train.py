import argparse
import os
import time
import json
import numpy as np
import random

from datasets import BasicDataset
from models import Pix2PixGenerator, CycleGANGenerator, CustomGeneratorA, CustomGeneratorB
from patchgan import PatchGAN
from utils import GANLoss, init_weights, save_model_params, wgangp_gradient_penalty

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform training of the polyp inpainting model")

    # Required arguments
    parser.add_argument("--dataset_dir", type=str, required=True, help="Location of polyp dataset")
    parser.add_argument("--save_loc", type=str, required=True, help="Directory to save model and training information")
    parser.add_argument("--model", type=str, required=True, help="Which generator model to use? (pix2pix, cyclegan, customA, customB)")

    # Optional arguments
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train for")
    parser.add_argument("--decay_epoch", type=int, default=100, help="Epoch to commence linear lr decay to 0 from")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate of the adam optimisers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size used for training")
    parser.add_argument("--batch_print_freq", type=int, default=100, help="Number of batches between printing training info (and logging training info too)")
    parser.add_argument("--use_batch_norm", action="store_true", default=False, help="Use batch normalisation instead of instance normalisation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loader")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="Use GPU for training (if available)")
    parser.add_argument("--init_mode", type=str, default="xavier", help="Weight initialisation mode (normal or xavier)")
    parser.add_argument("--GAN_loss", type=str, default="vanilla", help="GAN loss formulation (vanilla, lsgan or wgangp)")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="Weight for the l1 reconstruction loss term - reduce if using lsgan")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Weight for WGAN-GP gradient penalty term (WGAN-GP only)")
    parser.add_argument("--num_disc_iterations", type=int, default=5, help="Number of critic (discriminator) iterations to train before training a generator iteration (WGAN-GP only)")

    opt = parser.parse_args()

    # If batch_print_freq is not a multiple of num_disc_iterations, make it so (since losses are only available for logging
    # on generator iterations) - only care about for WGAN-GP
    if opt.GAN_loss == "wgangp":
        if opt.batch_print_freq < opt.num_disc_iterations:
            opt.batch_print_freq = opt.num_disc_iterations
        else:
            opt.batch_print_freq = opt.batch_print_freq - (opt.batch_print_freq % opt.num_disc_iterations)

    print(opt)

    if opt.use_batch_norm:
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = nn.InstanceNorm2d

    # Sanity check arguments
    TRAIN_DIR = os.path.join(opt.dataset_dir, "train")
    assert os.path.isdir(TRAIN_DIR), "Cannot find training directory at "+str(TRAIN_DIR)
    assert os.path.isdir(os.path.join(TRAIN_DIR, "composites")), "Cannot find training composite images at "+str(os.path.join(TRAIN_DIR, "composites"))
    assert os.path.isdir(os.path.join(TRAIN_DIR, "polyps")), "Cannot find training polyp images at "+str(os.path.join(TRAIN_DIR, "polyps"))

    VAL_DIR = os.path.join(opt.dataset_dir, "val")
    assert os.path.isdir(VAL_DIR), "Cannot find validation directory at "+str(VAL_DIR)
    assert os.path.isdir(os.path.join(VAL_DIR, "composites")), "Cannot find validation composite images at "+str(os.path.join(VAL_DIR, "composites"))
    assert os.path.isdir(os.path.join(VAL_DIR, "polyps")), "Cannot find validation polyp images at "+str(os.path.join(VAL_DIR, "polyps"))

    # If an output directory already exists append "-new" to the save dir and try again
    while os.path.isdir(opt.save_loc):
        opt.save_loc = opt.save_loc+"-new"
    print("Creating output directory at %s" % opt.save_loc)
    os.mkdir(opt.save_loc)
    TRAINING_SAMPLES_DIR = os.path.join(opt.save_loc, "training_samples")
    os.mkdir(TRAINING_SAMPLES_DIR)

    # Save training arguments as json file in model output folder for reference
    opt_dict = vars(opt)
    with open(os.path.join(opt.save_loc, "training_options.json"), 'w') as fp:
        json.dump(opt_dict, fp, sort_keys=True, indent=4)

    # Calculate a multiplier to apply to lr at any epoch to implement the linear
    # lr decay to 0 starting at decay_epoch
    def lr_function(epoch):
        return min(opt.lr, (opt.lr - (epoch - opt.decay_epoch + 1)*(opt.lr/(opt.num_epochs-opt.decay_epoch + 1))))/opt.lr

    if opt.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("Using device :", device)

    # Training dataset
    train_data = BasicDataset(TRAIN_DIR, jitter=True, mirror=True)
    print("Training on %d images" % len(train_data))

    val_data = BasicDataset(VAL_DIR, jitter=False, mirror=False)
    print("Validation set: %d images" % len(val_data))

    # Training dataloader
    train_loader = DataLoader(
        train_data,
        batch_size = opt.batch_size,
        shuffle = True,
        num_workers= opt.num_workers, 
    )

    # loss function definitions
    loss_adversarial = GANLoss(mode=opt.GAN_loss).to(device)
    loss_reconstruction = torch.nn.L1Loss().to(device)

    # Initialise models
    if opt.model == "pix2pix":
        generator = Pix2PixGenerator(norm_layer=norm_layer).to(device)
    elif opt.model == "cyclegan":
        generator = CycleGANGenerator(norm_layer=norm_layer).to(device)
    elif opt.model == "customA":
        generator = CustomGeneratorA(norm_layer=norm_layer).to(device)
    elif opt.model == "customB":
        generator = CustomGeneratorB(norm_layer=norm_layer).to(device)
    else:
        raise Exception("The model %s is not recognised" % opt.model)
    init_weights(generator, mode=opt.init_mode)
    discriminator = PatchGAN(norm_layer=norm_layer).to(device)
    init_weights(discriminator, mode=opt.init_mode)

    # print model details
    print("Generator architecture:")
    for child in generator.children():
        print(child)
    gen_param_count = sum([param.numel() for param in generator.parameters()])
    print("Generator parameters: %.3f M" % (gen_param_count / 1e6))

    print("Discriminator architecture:")
    for child in discriminator.children():
        print(child)
    disc_param_count = sum([param.numel() for param in discriminator.parameters()])
    print("Discriminator parameters: %.3f M" % (disc_param_count / 1e6))

    # optimisers
    G_optimiser = torch.optim.Adam(generator.parameters(), opt.lr, betas=(0.5, 0.999))
    D_optimiser = torch.optim.Adam(discriminator.parameters(), opt.lr, betas=(0.5, 0.999))
    # schedulers (for linearly decreasing lr from lr->0 starting at decay epoch):
    G_scheduler = torch.optim.lr_scheduler.LambdaLR(G_optimiser, lr_lambda=lr_function)
    D_scheduler = torch.optim.lr_scheduler.LambdaLR(D_optimiser, lr_lambda=lr_function)

    
    # numpy array to log the losses
    required_history_size = (len(train_loader) * opt.num_epochs) // opt.batch_print_freq
    losses_history = np.zeros(shape=(required_history_size, 4)) # index 0: batches, 1: D_loss, 2: G_adv, 3: G_recon

    total_batches = 0 # total-batch counter
    generator.train()
    discriminator.train()
    print("\n--- Beginning Training ---")
    for epoch in range(opt.num_epochs):
        epoch_start = time.time() # per-epoch timer
        for i, batch in enumerate(train_loader):
            iteration_start = time.time() # per-iteration (batch) timer
            total_batches += 1
            
            composites = batch[0].to(device)
            polyps = batch[1].to(device)
            # generate the fake images
            fake_polyps = generator(composites)
            
            # --- Discriminator  ---
            # Need to re-enable gradient calculations of D (see generator comment)
            for param in discriminator.parameters():
                param.requires_grad = True
                
            D_optimiser.zero_grad()
            # fake images - DETACH FAKE IMAGES TO AVOID BACKPROP TO THE GENERATOR
            pred_fake = discriminator(composites, fake_polyps.detach())
            gan_loss_fake = loss_adversarial(pred_fake, False)
            # real images
            pred_real = discriminator(composites, polyps)
            gan_loss_real = loss_adversarial(pred_real, True)
            
            # combine and update
            if opt.GAN_loss == "wgangp":
                # also note to self, may need to separate the loss_D.backwards() and do the gradient_penalty.backwards()
                # with retain_graph=True
                gradient_penalty = wgangp_gradient_penalty(discriminator, composites, polyps, fake_polyps, device, lambda_gp=opt.lambda_gp)
                loss_D = gan_loss_fake + gan_loss_real + gradient_penalty
                loss_D.backward(retain_graph=True)
            else:
                loss_D = 0.5*(gan_loss_fake + gan_loss_real)
                loss_D.backward()
            
            D_optimiser.step()
            
            
            # --- Generator ---
            # When using WGAN-GP, critic (D) trains for multiple iterations per training iteration on G
            if (opt.GAN_loss != "wgangp") or (total_batches % opt.num_disc_iterations == 0):
                # When updating generator do not need gradients of D, so for efficiency sake disable gradient calculations
                for param in discriminator.parameters():
                    param.requires_grad = False
                
                G_optimiser.zero_grad()
                # adversarial loss
                pred_fake = discriminator(composites, fake_polyps)
                gan_loss_G = loss_adversarial(pred_fake, True) # Loss computed with fake output but real label => generator wants to trick D
                # L1 loss
                loss_L1 = loss_reconstruction(fake_polyps, polyps) * opt.lambda_l1
                
                # combine and update
                loss_G = gan_loss_G + loss_L1
                loss_G.backward()
                G_optimiser.step()
                
                if (total_batches % opt.batch_print_freq == 0):
                    # print losses
                    print(
                        "Epoch: %d/%d, Batch: %d/%d, Batch Time: %.3fs, D_loss: %.3f, G_loss: %.3f [G_adv: %3f, G_recon: %3f]"
                        % (
                            epoch, opt.num_epochs,
                            i+1, len(train_loader),
                            time.time() - iteration_start,
                            loss_D.item(),
                            loss_G.item(),
                            gan_loss_G.item(),
                            loss_L1.item()
                        )
                    )
                    
                    # log losses in numpy array
                    losses_history[(total_batches//opt.batch_print_freq)-1] = [total_batches, loss_D.item(), gan_loss_G.item(), loss_L1.item()]
            
            
        # End of epoch tasks
        lr_prev = G_scheduler.get_last_lr()[0]
        G_scheduler.step()
        D_scheduler.step()
        lr_new = G_scheduler.get_last_lr()[0]
        print("Lr %.6f -> %.6f" % (lr_prev, lr_new))
        # sanity check both schedulers have the same lr
        assert lr_new == D_scheduler.get_last_lr()[0]
        
        print("Epoch %d completed in %.3fs\n" % (epoch, time.time() - epoch_start))

        # At the end of every epoch, generate and save an image from the validation set
        sample_idx = random.randrange(len(val_data))
        composite, polyp = val_data[sample_idx]
        generated = generator(composite[None, ...].to(device))
        combined = torch.cat([composite, generated[0, ...].cpu(), polyp], dim=2)
        img = T.ToPILImage()((combined+1)/2) # de-normalize ([-1,1] -> [0,1])
        img.save(os.path.join(TRAINING_SAMPLES_DIR, "epoch"+str(epoch+1)+".jpg"))


        
    print("--- Training complete ---")

    save_model_params(generator, os.path.join(opt.save_loc, "generator.pth"))
    save_model_params(discriminator, os.path.join(opt.save_loc, "discriminator.pth"))
    print("Models saved")

    np.save(os.path.join(opt.save_loc, "loss_history"), losses_history)
    print("Loss history saved")

            