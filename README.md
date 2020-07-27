Predict objects in images
----------------

The code in `predict_objects_in_images.py` predicts objects in a given image with trained CNNs. The program works with MobileNet and ResNet50-models.

Setup
----------------

You will need a trained CNN in form of a h5-file as well as a .npy-file where the class indices of that model are stored.
* .h5-file: Generated e.g. with tf.keras.models.save_model
* .npy-file: Generated e.g. with np.save(PATH, ImageDataGenerator.flow_from_directory.class_indices)


Usage
----------------

Call the function `predict_objects()` with the following arguments:

* Path to the image
* Path to the .h5-file
* Path to the .npy-file

Optional:
* Kind of model ("resnet" or "mobilenet"; default="resnet")
* Confidence threshold: How confident should the model be before returning detected objects (0.0 - 1.0; default=0.7)

Usage in bash
----------------

Example for usage in bash:

    $ python predict_objects_in_images.py PATH_TO_IMAGE PATH_TO_H5_MODEL PATH_TO_NPY_CLASS_INDICES
