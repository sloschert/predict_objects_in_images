"""
Predict objects in an image with trained cnn-models. Works with MobileNet and
ResNet50-models. Works also in bash.
Example for usage in bash:
    python predict_objects_in_images.py example_img.jpg\
    model_resnet50_29classes.h5 resnet50_29classes.npy

"""

import sys
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

def predict_objects(user_img, h5_model, npy_class_indices,\
 kind_of_model="resnet", conf_threshold=0.7):
    """
    INPUT:
    path to image
    h5-model
    npy-class_indices
    kind of model: "resnet" or "mobilenet"
    conf_threshold: how confident should the model be before returning detected
    model. float 0.0 - 1.0.

    OUTPUT:
    list with predicted objects
    """
    def rotate_image(im):
        """
        Rotates image if necessary.
        """
        print("im size", im.size)

        # find if image position is horizontal or vertical
        try:
            exif = im._getexif()
            if exif[274] == 6:
                print("exif 6")   # flipped to left
                rotate_img = True
            elif exif[274] == 1:
                print("exif 1")
                rotate_img = False
            else:
                print("neither exif 6 nor 1")

        except AttributeError:
            print("Could not get exif - Bad image!")
        except TypeError:
            print("Could not get exif - Bad image!")

        (width, height) = im.size
        if not exif:
            if width < height:
                print("w < h, vertical pic")
                rotate_img = False
            else:
                print("w > h, horizontal pic")
                rotate_img = True

        print("rotate_image= ", rotate_img)

        ## rotate image
        if rotate_img:
            im = im.transpose(Image.ROTATE_270)

        return im

    def reshape_image(im):
        """
        Resize image (differently if vertical or horizontal)
        """
        if im.shape[0] > im.shape[1]:
            newHeight = 1066    # was originally 200
            newWidth = int(im.shape[1]*1066/im.shape[0])
            im = cv2.resize(im, (newWidth, newHeight))
        else:
            newHeight = 600    # was originally 200
            newWidth = int(im.shape[1]*600/im.shape[0])
            im = cv2.resize(im, (newWidth, newHeight))
        return im

    def selective_search(im, quality):
        """
        Run selective search on image and return rectanculars (boxes).
        """
        # speed-up using multithreads
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)

        # create Selective Search Segmentation Object using default parameters
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        # set input image on which we will run segmentation
        ss.setBaseImage(im)

        # Switch to fast but low recall Selective Search method
        if quality == 'f':
            ss.switchToSelectiveSearchFast()

        # Switch to high recall but slow Selective Search method
        elif quality == 'q':
            ss.switchToSelectiveSearchQuality()

        print("run selective search on image")
        rects = ss.process()
        return rects

    def get_parts_of_image(rects, im):
        """
        Generates images out of rects.
        """
        parts_of_image = []
        for i in rects:
            imC = im.copy()
            x, y, w, h = i
            part_of_image = imC[y:y+h, x:x+w]

            # only add to parts_of_image, if w and h large enough to later expand
            # to 224x224 AND if ratio is between certain boundaries
            if w > 75 and h > 75 and 0.5 < w/h < 2:
                parts_of_image.append(part_of_image)

        print("Expanding clips to 224x224")

        parts_of_image_resized = []
        for i in parts_of_image:
            parts_of_image_resized.append(cv2.resize(i, (224, 224)))

        assert len(parts_of_image_resized) == len(parts_of_image)
        print(f" I have {len(parts_of_image_resized)} clips/boxes")

        parts_of_image_resized = np.array(parts_of_image_resized)
        return parts_of_image_resized

    def search_dic_by_value(value, class_indices)->str:
        """
        searches the "class indices" for a value (int) and gives back the
        corresponding key (str).
        """
        for i in class_indices.items():
            if i[1] == value:
                return i[0]

    print("kind of model: ", kind_of_model)
    print("threshold: ", conf_threshold)
    print("image: ", user_img)

    # load an h5-model
    model = load_model(h5_model)
    print("model: ", h5_model)

    # load corresponding class indices
    class_indices = np.load(npy_class_indices, allow_pickle=True).item()

    # specify on how many boxes you want to make a prediction
    max_pred = 30
    print(f"Predictions on {max_pred} boxes")

    # selective search-quality: f: fast but low recall, q: high recall but slow
    quality = "f"

    im = load_img(user_img)
    im = rotate_image(im)
    im = img_to_array(im)
    im = im.astype(np.uint8)

    print("shape before resizing: ", im.shape)

    im = reshape_image(im)

    print("shape after resizing: ", im.shape)

    rects = selective_search(im, quality)

    parts_of_image_resized = get_parts_of_image(rects, im)

    print("Start to (preprocess and) predict")

    if kind_of_model == "mobilenet":
        parts_of_image_resized_preprocessed = preprocess_input(parts_of_image_resized)
        predictions = model.predict(parts_of_image_resized_preprocessed[:max_pred])
    elif kind_of_model == "resnet":
        predictions = model.predict(parts_of_image_resized[:max_pred])

    # add confident predictions to a list, in the format of a tuple:
    # (Name of Object, confidence score)

    print("Create list with confident predictions")
    list_of_predictions = []
    for pred in predictions:
        if max(pred) > conf_threshold:
            # append tuple to prediction list
            if  'Noise' in search_dic_by_value(pred.argmax(), class_indices):
                list_of_predictions.append\
                ((f"___ Only Noise ({conf_threshold})", None))
            else:
                list_of_predictions.append((search_dic_by_value\
                (pred.argmax(), class_indices), max(pred)))
        else:
            # append tuple to list
            list_of_predictions.append((f"___ Under confidence threshold \
            ({conf_threshold})", None))

    # create set of confidently detected objects
    detected_objects = set()
    for i in list_of_predictions:
        if i[1]:
            detected_objects.add(i[0])

    print("I have detected the following objects: \n\n", detected_objects)
    return list(detected_objects)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        predict_objects(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        predict_objects(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 6:
        predict_objects(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], \
        sys.argv[5])
