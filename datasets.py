import os
import random

from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms 
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]
def has_image_extension(file):
    return any(file.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_names(path):
    image_names = []
    
    for _, _, names in sorted(os.walk(path)):
        for name in names:
            if has_image_extension(name):
                image_names.append(name)
                
    return image_names

# Assuming different data format than before, root folder simply contains two folders, 'composites' and 'polyps'
# containing the image composites (mask overlayed over image) and polyps, with corresponding images in each folder
class BasicDataset(Dataset):
    def __init__(self, path, jitter=False, mirror=False):
        self.jitter = jitter
        self.mirror = mirror
        self.composite_path = os.path.join(path, "composites")
        self.polyp_path = os.path.join(path, "polyps")
        self.image_names = get_image_names(self.composite_path)
        self.num_images = len(self.image_names)
        
        # sanity check - ensure each folder has the same number of images:
        assert self.num_images == len(get_image_names(self.polyp_path))
        # check that each image in the composites folder also appears in the polyps folders:
        for name in self.image_names:
            assert os.path.isfile(os.path.join(self.polyp_path, name))
        
    def __len__(self):
        return self.num_images
    
    def transform(self, composite, polyp):
        # Jitter as described in pix2pix paper - resize to 286x286 then random crop to 256x256
        # Awkwardness arises from having to do the SAME crop to both images
        # Also apply random mirroring too
        if self.jitter or self.mirror:
            if self.jitter:
                resize = transforms.Resize((286, 286))
                composite = resize(composite)
                polyp = resize(polyp)
            
                # Get random crop parameters, then apply the same crop to both images
                i, j, h, w = transforms.RandomCrop.get_params(
                    composite, output_size=(256, 256))
                composite = TF.crop(composite, i, j, h, w)
                polyp = TF.crop(polyp, i, j, h, w)
            
            else:
                resize = transforms.Resize((256, 256))
                composite = resize(composite)
                polyp = resize(polyp)
                
                
            if self.mirror:
                # Random horizontal flipping
                if random.random() > 0.5:
                    composite = TF.hflip(composite)
                    polyp = TF.hflip(polyp)

                # Random vertical flipping
                if random.random() > 0.5:
                    composite = TF.vflip(composite)
                    polyp = TF.vflip(polyp)

            composite = TF.to_tensor(composite)
            polyp = TF.to_tensor(polyp)
            
            # normalise results to -1, 1 (ganhack)
            composite = TF.normalize(composite, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            polyp = TF.normalize(polyp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
        # No jitter case is simple - just return the image tensor after resizing to 256x256
        else:
            composed_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalisation ganhack
            ])
            composite = composed_transforms(composite)
            polyp = composed_transforms(polyp)
        
        return composite, polyp
        
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        composite_impath = os.path.join(self.composite_path, image_name)
        polyp_impath = os.path.join(self.polyp_path, image_name)
        
        composite_image = Image.open(composite_impath).convert("RGB")
        polyp_image = Image.open(polyp_impath).convert("RGB")
        
        composite_image, polyp_image = self.transform(composite_image, polyp_image)
        
        return (composite_image, polyp_image)