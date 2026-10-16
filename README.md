This Repository Contains a collection of models which aim to create synthetic MRI images of human brains. These models' are then compared against each other via SSIM, PSNR, and MSE measures.

Input: HD5F files of MRI Modality 1 and MRI Modality 2
Output: SSIM, PSNR, MSE of each model's results compared against each other in a table

Currently, these models are all using Brats 2021 Data

Models which require training on Brats Data:
cycleGAN, Pix2PixHD, UNIT, MUNIT

Models which require testing on Brats data:
Clasp_2D, cycleGAN, Pix2PixHD, UNIT, MUNIT

Currently, Train, validation, test split is as follows:
Train: 0.8 unpaired data
Validation: 0.1 paired data
Test: 0.1 paired data

Early stopping utilizes L1 loss, comparing generated images to ground truth

Currently, only translation from T1.post to T2 has been implemented
