# CNCL_Medical_Image_Denoising

Pytorch implementation of **Content-Noise Complementary Learning for Medical Image Denoising**.

## File Description

### 1. code

**DataLoader_train.py**: python script to build pytorch Dataset and DataLoader for training.

**DataLoader_test.py**: python script to build pytorch Dataset and DataLoader for test.

**model.py**: the implementation of our proposed CNCL-U-Net achitectures.

**train.py**: a basic template python file for training the model.

**test.py**: a basic template python file for testing the model.

### 2. data

**clean**: clean medical images (i.e., content images), such as full-dose CT images.

**noisy**: noisy medical images (i.e., noise corrupted-images), such as low-dose CT images.

**dataset_division**: contain four .txt files, storing the file names of the images in the training set or test set, respectively.

### 3. result

**saved_models**: save the trained models.
**test**: save the predicted content images when testing.
