## Grad-Cam Visualizations

These scripts will enable gradcam saliency maps to be generated. The output will be all images in the test set and corresponding saliency maps. Thus, you will have image pairs that you should overlay using the overlay.py script.

Read gradcam paper [here](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf).

These scripts are meant to be run from the same directory as dataset.py. Depending on what model you are loading to visualize, you will also need to alter the model in visualize\_image\_model.py. This script takes the mean of all test images by class, adds gaussian blur to each image (1 per class), and then maximizes the model's prediction for each class given the input image. Because this is a random process (generating the input images), the default is to load 2 image models and maximize the classifications for both, so you can compare.

To run:

	python visualize_image_model.py --load-model=/path/to/weights.ckpt --load-model2=/path/to/other/weights.ckpt --image-dir=/path/to/h5

To overlay images and their saliency maps (default output filename: severe.png):

	python overlay.py xray.png saliency.png 
