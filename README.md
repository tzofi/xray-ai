# Chest X-Ray Image Models

PyTorch Implementations for:
 - Supervised baseline
 - Self-supervised learning (rotation + shift prediction)
 - SimCLR (Simple Framework for Contrastive Learning)
 - Gradcam saliency maps (for interpretability)
 - Adversarial perturbations for maximizing classifications

Contrastive directory includes a contrastive learning framework for additional contrastive methods, such as MOCO, PIRL, InfoMin, CMC, etc. These methods train (much) faster than SimCLR, however, experiments have indicated SimCLR achieves higher performance on the x-ray datasets.  These methods also fully support Distributed Data Parallel (DDP) and multi-precision, for extremely fast training in PyTorch.

## Setting up your environment
This code was developed with PyTorch 1.4.0. For all packages, on a local Linux machine, install Anaconda and run:

	conda env create -f environment.yml

Then:

	conda activate ssl


## Dataset Info
To ensure quick and efficient training, we resize and pad x-ray images before training, and save the output images in a single HDF5 file. This is done using the CXRResizer class in dataset.py. To create a dataset, you need a directory of NPY files and a CSV with label information. Then:

	dataset = CXRresizer("labels.csv", "/path/to/npy", size=(2048,2048))
	dataset.transfer("2048_dataset.h5")

## Run Supervised Learning
Train the classifier model.

    python supervised/train.py --image-dir=/path/to/h5

Test the classifier model.

    python supervised/test.py --load-model=models/best.ckpt --image-dir=/path/to/h5

## Run Self-Supervised Learning (SSL)
Train the feature extracting model with self-supervised learning (i.e. image rotation task).

    python ssl/train_features.py --batch-size=64 --accumulation-steps=4 --tau=0.5 
                              --feature-size=256 --image-dir=/path/to/h5
    
Train the classifier model. Needs a saved feature model to extract features from images. Uses 1 fully connected.

    python ssl/train_classifier.py --load-model=models/best.ckpt --image-dir=/path/to/h5

## Run SimCLR
Train the feature extracting model (uses ResNet18-CIFAR)

    python simclr/train_features.py --batch-size=64 --accumulation-steps=4 --tau=0.5 
                              --feature-size=256 --image-dir=/path/to/h5
    
Train the classifier model. Needs a saved feature model to extract features from images. Uses 1 fully connected.

    python simclr/train_classifier.py --load-model=models/best.ckpt --image-dir=/path/to/h5

## Interpretability Using Gradcam
Visit the README in the gradcam directory.

## Robustness
Visit the README in the adversarial directory. This library allows you to synthesis images from a seed image that maximizes the classification confidence of your model. This can give insight into relevant learned features. One might ask, does the generated image correspond with what a doctor would expect for the maximized classification?
