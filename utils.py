import torch
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as autograd

# Wrapper for the GAN/adversarial loss, as the PatchGAN produces a shaped output (typically 1x30x30 when given
# 256x256 input) which needs to be averaged over. The wrapper abstracts away the target label tensor which does this
# Mode copies the pix2pix functionality: vanilla does normal adversarial GAN loss whilst lsgan does LSGAN loss
class GANLoss(nn.Module):
    def __init__(self, mode="vanilla", fake_label=0.0, real_label=1.0):
        super(GANLoss, self).__init__()
        self.mode = mode
        if mode == "vanilla":
            # BCEWithLogitsLoss applies a sigmoid layer before computing BCELoss, which covers
            # the regular GAN loss formulation (reduction = mean by default)
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            # for lsgan no sigmoid activation is applied, and instead the MSE is computed over
            # the entire discriminator output compared against the ground truth target tensor
            # WARNING: if using this change the l1_lambda accordingly as it has a smaller range
            # compared to vanilla loss (lambda=100 down to 10)
            self.loss = nn.MSELoss()
        elif mode == "wgangp":
            # WGAN-GP loss has to be handled differently - firstly in __call__ the loss
            # returned is simply the discriminator mean (-ve for real examples, +ve
            # for fake examples) to provide the critic loss, and then the gradient
            # penalty calculation must be called and combined externally (in training code)
            self.loss = None
        else:
            raise Exception("The gan loss mode %s is not recognised" % mode)
        
        # buffers for GT tensors with shape matching the output of the discriminator
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
    
    # Create a GT tensor with shape matching that of shape_tensor, with GT label determined by target_is_real
    def GT_like(self, shape_tensor, is_real):
        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(shape_tensor)
        
    # Given the tensor output of the discriminator (typically 1x30x30 when using 70x70 patchgan on 256x256 images)
    # compute the loss. In both cases (vanilla and lsgan) a ground truth tensor of matching shape will be required
    # is_real is a flag to determine whether the GT should be set to the real or fake label
    def __call__(self, disc_output, is_real):
        if self.mode == "wgangp":
            if is_real:
                loss = -disc_output.mean()
            else:
                loss = disc_output.mean()
        else:
            GT_tensor = self.GT_like(disc_output, is_real)
            loss = self.loss(disc_output, GT_tensor)
        return loss

# Calculate the gradient penalty term used for WGAN-GP
# device argument is necessary as unlike GANLoss which is a nn.Module and can be sent to the device, this function is not
def wgangp_gradient_penalty(discriminator, composites, real_tensor, fake_tensor, device, lambda_gp=10.0):
    # first need to create the x_hats, aka the points sampled between the real and fake data samples via straight line
    # interpolation between each sample pair using a random alpha between 1 and 0.
    N, C, H, W = real_tensor.shape # batch size
    alpha = torch.rand(N, 1, 1, 1, device=device)
    alpha = alpha.expand(N, C, H, W)
    interpolated = alpha*real_tensor + ((1-alpha)*fake_tensor)
    
    # set requires grad on interpolated, since by default the training loop disables gradient calculations of
    # discriminator during generator update phase
    interpolated.requires_grad_(True)
    disc_interpolated = discriminator(composites, interpolated)
    
    gradients = autograd.grad(outputs=disc_interpolated, inputs=interpolated,
                             grad_outputs = torch.ones(disc_interpolated.size()).to(device),
                             create_graph=True, retain_graph=True, only_inputs=True)
    
    # Note - may want to compute the norm manually and add an epsilon before the sqrt to avoid
    # problems with derivatives close to 0
    gradient_penalty = ((gradients[0].norm(2, dim=(1,2,3)) - 1)**2).mean() * lambda_gp
    return gradient_penalty


# Weight initialisation
def init_weights(model, gain=0.02, mode="normal"):
    def init_inner(m):
        classname = m.__class__.__name__
        # initialising convolutional layers:
        if hasattr(m, "weight") and (classname.find("Conv") != -1):
            if mode == "xavier":
                init.xavier_normal_(m.weight.data, gain)
            elif mode == "normal":
                init.normal_(m.weight.data, 0, gain)
            else:
                raise Exception("Weight initialisation mode %s is not recognised" % mode)
            
            # some conv layesr have biases:
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # initialising batchnorm layers
        elif (classname.find("BatchNorm2d") != -1):
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_inner)
        
# Helper functions to save and load models, best practice is to save and load using the state_dict (only
# serialising the parameters, not the whole class)
def save_model_params(model, path):
    torch.save(model.cpu().state_dict(), path)

def load_model_params(model, path):
    model.load_state_dict(torch.load(path))