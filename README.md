# UCL-Project
Code used as part of my UCL masters dissertation project.

Overview:
- ``datasets.py``: Contains the pytorch dataloader used to load the data.
- ``models.py``: Defines the various generator architectures used in the project.
- ``patchgan.py``: Defines the PatchGAN discriminator architecture used.
- ``train.py``: Trains the conditional GAN for polyp synthesis. For usage run ``python train.py -h``.
- ``utils.py``: Defines various util functions used in the project.

Misc folder contains an example training output folder (`customB`) as well as a the python notebook used to analyse a model output folder and a loading script to create the training dataset.
Note that the dataset loading script requires the CVC-ClinicDB zip folder to be preprocessed to only contain JPG files, not the default TIF files. It may be easier to only provide the kvasir-seg zip folder and just train on the 1000 images from that.
