# Image_Orientation
Objectives Apply transfer learning to automatically detect and correct orientation of an image.

Modified Indoor CVPR dataset images by rotating the images into 0, 90, 180, 270 degrees separately to build a new dataset. Extracted features via VGG16 network pre-trained on ImageNet and save features into hdf file. Trained a logistic regression classifier built on top of the VGG16 to correct orientation classifier and evaluate the model. Defined an end-to-end pipeline so that we can input an image and its orientation will be corrected.

Packages Used

Python 3.6

OpenCV 4.0.0

keras 2.1.0

Tensorflow 1.13.0

cuda toolkit 10.0

cuDNN 7.4.2

NumPy

SciPy

Approaches

The dataset used in the project is Indoor CVPR (reference). The dataset contains 15620 total images, which has 67 indoor room/scene categories, including homes, offices, public spaces, stores, and etc.

Build dataset The create_dataset.py (check here) is responsible for randomly (uniformly) rotating images either by 0 (no change), 90 degrees, 180 (flipped vertically) degrees, or 270 degrees. Thus, there are four categories, each having about 3600 images per angle.

Extract features The extract_features.py (check here) is responsible for extracting features via VGG16 network pre-trained on ImageNet.

Here is a helper function:

The hdf5datasetwriter.py (check here) under pipeline/io/ directory, defines a class that help to write raw images or features into HDF5 dataset.

Train and evaluate the logistic regression classifier The train_model.py (check here) is responsible for training and evaluating the logistic regression classifier, building on top the VGG16 network.

End-to-end orientation correction pipeline The orient_images.py (check here) builds an end-to-end orientation correction pipeline, which we can input an image and its orientation will be corrected accordingly.
