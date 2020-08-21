# The X-Ray AI (XAI) Toolbox

PyTorch Implementations for:
 - Supervised baseline
 - Self-supervised learning (rotation + shift prediction)
 - SimCLR (Simple Framework for Contrastive Learning)
 - PyContrast framework, including MOCO, PIRL, InfoMin, CMC, etc.
 - Gradcam saliency maps (for interpretability)
 - Adversarial perturbations for maximizing classifications

Contrastive directory includes a contrastive learning framework for additional contrastive methods, such as MOCO, PIRL, InfoMin, CMC, etc. These methods train (much) faster than SimCLR, however, experiments have indicated SimCLR achieves higher performance on the x-ray datasets.  These methods also fully support Distributed Data Parallel (DDP) and multi-precision, for extremely fast training in PyTorch.

NOTE for MIT LL: This repo and all associated data, documents, and presentations are stored on LLGRID in the RL4MD\_shared directory. The raw NPY image files are stored on lambda-stack in /mnt/USB/npy.

## Setting up your environment
This code was developed with PyTorch 1.4.0. For all packages, on a local Linux machine, [install Anaconda](https://docs.anaconda.com/anaconda/install/linux/) and run:

	conda env create -f environment.yml

Then:

	conda activate ssl

If you are developing on the MIT LL internal LLSC Grid, you should follow a separate set up procedure. First, load the necessary compute resources for an interactive session (you may also launch a batch script, but will need to create it yourself):

	LLsub -i -g tesla:4 // OR
	LLsub -i -g volta:2 // if you use a volta GPU, please then run: source /etc/profile
	
Next, load the needed modules:

	module load anaconda2020b cuda-10.1
	
Next, load all proxies:

	module load proxy-mitll

Please refer to LLSC for help installing all Python packages. [This](https://lldrpl1prod.llan.ll.mit.edu/llsc/installing-python-packages) is their current documentation, which instructs you to install each package not provided by the module via pip:

	pip install --user torchlars // OR, if you are having issues with the proxy:
	pip install --proxy http://llproxy.llan.ll.mit.edu:8080 --user torchlars
	
Rather than installing packages one by one, you can try [these instructions](https://lldrpl1prod.llan.ll.mit.edu/llsc/installing-python-packages):

	conda env create --prefix /home/gridsan/<MyUserName>/.conda/envs/ssl -f environment.yml
	
At this point, you can procede with the following instructions without issue.

## Dataset Info
We utilize the [MIMIC CXR Dataset](https://physionet.org/content/mimic-cxr/2.0.0/). Labels were extracted using keyword matching (NegEx method). MIT campus also has a "gold" standard dataset of manually labeled x-rays. Please contact Polina Golland or Ruizhi Liao for access.

To ensure quick and efficient training, we resize and pad x-ray images before training, and save the output images in a single HDF5 file. This is done using the CXRResizer class in dataset.py. To create a dataset, you need a directory of NPY files and a CSV with label information. Then:

	dataset = CXRresizer("labels.csv", "/path/to/npy", size=(2048,2048))
	dataset.transfer("2048_dataset.h5")

For work being done at MIT Lincoln Laboratory, all raw data is on LLGRID in: RL4MD_shared/data. Several HDF5 files have already been created and lie in the data directory. 512_dataset.h5 is labeled data only (~7300 images). 512_full_dataset.h5 is all data, including unlabeled data. The raw npy files are on the lambda-stack machine at this path: /mnt/USB/npy. If you are unable to find it, you may need to mount the drive: sudo mount /dev/sdc1 /mnt/USB. You can then use this data to create new H5 files using the CXRresizer.

Labels are on LLGRID at this path: RL4MD_shared/data/edema_labels-12-03-2019/mimic-cxr-sub-img-edema-split-allLabeledCXR.csv. The labels are in this format:

	subject_id,study_id,dicom_id,edeme_severity,fold
	14003369,57020861,4523640b-e402e256-094ad3c4-f6d6e0f3-9fb696fe,1,1
	18230892,54172360,956b6351-eeb640bd-fc587be6-0aec3312-64cd46f2,0,2
	18230892,56148624,62efec44-fb117321-2cf76351-1803dcf1-b3049137,2,2
	18230892,56332957,08af41b7-8516e172-ecb75055-79a9de27-b23e5670,2,2
	18527164,54888772,68c157cf-49158016-2d406fa8-e9752a35-02d8cda1,3,3
	18527164,56201635,4432cf56-ee25936a-768c129b-1e0e83f8-a94ff072,3,3

dicom_id corresponds with the filename of the npy x-ray images. edeme_severity is the edema severity and ranges from 0 (none) to 3 (severe). Labels can be encoded ordinally or as a one-hot encoding. We use one-hot. Fold is the data fold, which can be 1, 2, 3, 4, 5, or TEST. Typically we include 4 folds in training, hold one out for validation, and always hold out TEST for testing.

The dataset.py contains PyTorch Dataset classes for supervised training, contrastive training, and self-supervised training. The CXRDataset class has no augementation by default. To enable, pass in do_transform=true when instantiating a dataset. Default augmentation is random crop and rotation, but more can be added.

For data augmentation, we rely on CV2 operations as this is MUCH faster than using torchvision augmentation.

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

![Example gradcam image for SimCLR model](https:github.com/tzofi/xray-ai/blob/master/gradcam/contrastive_learning_gradcam.png)

## Robustness
Visit the README in the adversarial directory. This library allows you to synthesis images from a seed image that maximizes the classification confidence of your model. This can give insight into relevant learned features. One might ask, does the generated image correspond with what a doctor would expect for the maximized classification?
