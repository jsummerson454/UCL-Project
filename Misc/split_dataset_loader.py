import os
import zipfile
import shutil
import cv2
import numpy as np
import random

# specify data split here
num_val = 62

# load dataset zip file
KVASIR_PATH = "./kvasir-seg.zip"
CVC_PATH = "./CVC-ClinicDB-NOTIFF.zip"

DATA_PATH = "split_dataset"
kvasir_zf = zipfile.ZipFile(KVASIR_PATH)
cvc_zf = zipfile.ZipFile(CVC_PATH)

# get all image paths
im_paths = [fn for fn in kvasir_zf.namelist() if fn.split(".")[-1] == "jpg"]
im_paths.extend([fn for fn in cvc_zf.namelist() if fn.split(".")[-1] == "jpg"])
num_images = int(len(im_paths)/2)

print("\nDataset contains %d image-label pairs." % num_images)

num_train = num_images - num_val

print("Extracting data into %d-%d for train-val split..." % (num_train, num_val))

# extract image pairs to respective folders - in this case root directories are train/test which contain 1550 and 62 images respectively
folders = [os.path.join(DATA_PATH, dn) for dn in ["train", "val"]]
composite_paths = [os.path.join(dn, "composites") for dn in folders]
polyp_paths = [os.path.join(dn, "polyps") for dn in folders]
if os.path.exists(DATA_PATH): # clean up existing dataset folder if one already exists
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)
for folder in folders:
    os.mkdir(folder)
for composite_folder in composite_paths:
    os.mkdir(composite_folder)
for polyp_folder in polyp_paths:
    os.mkdir(polyp_folder)

# shuffle im_paths with fixed seed so that the other data loading scripts (for pix2pix) have the same train/test split
# also trimming to only contain the mask image paths, then loading the GI images on the fly
im_paths = [path for path in im_paths if path.startswith("Kvasir-SEG/masks") or path.startswith("Ground Truth")]
assert num_images == len(im_paths), "Number of masks does not match the number of images"
random.Random(15).shuffle(im_paths)
for i, path in enumerate(im_paths):
    if i%50 == 0:
        print("Processing image %d/%d" % (i, num_images))

    # train or val
    if i < num_train:
        fidx = 0
    else:
        fidx = 1

    # which zip folder
    if path.startswith("Kvasir-SEG"):
        zf = kvasir_zf
        source = "kvasir"
    else:
        zf = cvc_zf
        source = "cvc"
    
    # determine where to save image (destination)
    image_dst = os.path.join(polyp_paths[fidx], str(i)+".jpg")
    mask_dst = os.path.join(composite_paths[fidx], str(i)+".jpg")

    # handling conditional case separately since it needs to be overlayed on the polyp image
    mask_data = zf.read(path)
    image_mask = cv2.imdecode(np.frombuffer(mask_data, np.uint8), 1)
    if source == "kvasir":
        polyp_data = zf.read(path.replace("masks", "images"))
    else:
        polyp_data = zf.read(path.replace("Ground Truth", "Original"))
    image_polyp = cv2.imdecode(np.frombuffer(polyp_data, np.uint8), 1)

    composite_image = cv2.add(image_mask, image_polyp)
    cv2.imwrite(mask_dst, composite_image)
    cv2.imwrite(image_dst, image_polyp)
        
print("Complete")