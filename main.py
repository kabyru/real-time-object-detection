#Object Detection with a Defined Model, Real Time Detection

#Written by Kaleb Byrum, with code taken by Google Research

#First, import the necessary packages...
import numpy as np
import os
import sys
import tensorflow as tf
import pathlib
import time

#To engage with the webcam
from _thread import *
import cv2
from imutils.video import VideoStream
import imutils as im

#To engage with pictures...
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image #This uses Pillow, the successor to original PIL
from IPython.display import display

#Import the compiled object detection model (should have pip site-package installed in venv.)
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#Patches to the program
utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

#Prepare the model for use.
#These are the model loaders. Because of Raytheon proxy and the love of my internet access I will be manually implementing these rather than use slick Internet functions.
#There are two. One for the object detection model, and another for the Image Segementation model.
#note: we may only use the Object Detection model, at the least. This is the custom trained one...

#This controls the threshold of object detection performance.
minThreshold = 0.5

def load_obj_detection_model():
    model_dir = os.getcwd() + "\\models\\petInference\\saved_model" #This needs to be redirected.
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model

def load_img_seg_model():
    model_dir = os.getcwd() + "\\models\\maskModel\\saved_model" #This needs to be redirected.

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model

#Next, load the LABEL MAP LOCATION...
#We'll keep this relative, but it should follow this pattern...
PATH_TO_LABELS = os.getcwd() + "\\models\\annotations\\label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#Next, load the image path of the test images.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path("models/images/test")
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
#Upon success, there should be a dump of data that shows all the paths to train images.
print(TEST_IMAGE_PATHS)

#Next, load the object detection model with the previously defined functions...
detection_model = load_obj_detection_model()
masking_model = load_img_seg_model()

#This wrapper function will call the model, and then cleanup the outputs.
#This is where this program will differ from previous iterations, we need to only pull away 
#the largest object detected!
def run_inference_for_single_image(model, image):
    image = np.asarray(image) #Converts the input image into a numpy matrix...
    #The image will then be converted into a tensor. It will be converted using 'tf.convert_to_tensor'
    input_tensor = tf.convert_to_tensor(image)
    #The model expects a bunch of images, so add an axis with 'tf.newaxis'
    input_tensor = input_tensor[tf.newaxis,...]

    #Run inference operations...
    output_dict = model(input_tensor)

    #All outputs are batches tensors
    #Convert to numpy arrays, and take index [0] to remove the batch dimension
    #We're only interested in the first num_detections

    num_detections = int(output_dict.pop('num_detections'))
    print("There are a total of " + str(num_detections) + " objects detected in this image.")

    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    #Detection classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    #Handle models with masks, should a masking model be provided. This will apply to masking model detections only.
    if 'detection_masks' in output_dict:
        #Reframe the bbox mask to the image size...
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'], image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy() #Guessing this converts it back into a numpy array to return
    
    #print(output_dict)
    return output_dict

#The function will run on each input image from a webcam or flie path and show the result.
def show_inference(model, image_path, fromDirectory=False, showWindow=False, saveImage=True, count=0):
    #If the image comes from a webcam, then it will come already as a NP array.
    if (fromDirectory == False):
        image_np = image_path
    else:
        #If it comes from a JPG image, it will need to be converted.
        image_np = np.array(Image.open(image_path))
    
    #Run the detection using the previously defined function...
    output_dict = run_inference_for_single_image(model, image_np)

    #Now, visualize the results of the object detection function...
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks = output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates = True,
        min_score_thresh=minThreshold,
        max_boxes_to_draw=1,
        line_thickness = 8)
    
    display(Image.fromarray(image_np))
    resultingImage = Image.fromarray(image_np)
    if (showWindow == True):
        if (fromDirectory == True):
            resultingImage.show()
            time.sleep(3)
            resultingImage.close()
            return image_np
        else:
            #resultingImage.show()
            return image_np
    if (saveImage == True):
        resultingImage.save(os.getcwd() + "\\results\\" + str(count) + ".png")
    
    return_box_coords(output_dict, image_np)

def return_box_coords(detectDict, image): #TO-DO: Will need to coordinate detection_classes with detection_boxes
    boxes = np.squeeze(detectDict['detection_boxes'])
    scores = np.squeeze(detectDict['detection_scores'])
    classes = np.squeeze(detectDict['detection_classes'])

    #Set a minimum threshold score, let's do 90%
    min_score_thresh = minThreshold
    bboxes = boxes[scores > min_score_thresh]
    cclasses = classes[scores > min_score_thresh]

    #Get the image size
    im_height = int(np.size(image, axis = 0))
    im_width = int(np.size(image, axis = 1))

    final_box = []
    count = 0
    for box in bboxes:
        ymin, xmin, ymax, xmax = box
        final_box.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])
        count = count + 1

    #Operations to determine the largest box of the three:
    largestArea = 0
    boxCount = 0
    largestClass = 0
    largestBoxNumber = 0
    for box in final_box:
        xmin, xmax, ymin, ymax = box
        length = xmax - xmin
        width = ymax - ymin
        area = length * width
        if (area > largestArea):
            largestBoxNumber = boxCount
            largestArea = area
            largestClass = cclasses[boxCount]
        boxCount = boxCount + 1
    
    #Correlate the largestClass value to labels
    if (largestClass != 0):
        largestLabel = category_index[largestClass]
        print("The largest bounding box is Box # " + str(largestBoxNumber) + " which is of Class: " + str(largestClass) + " which correlates to Label: " + str(largestLabel))
        print("Extracted name: " + str(largestLabel['name']))
    else:
        print("(line 229) Error! No boxes detected! largestClass is still 0.")


#This handler will direct each new webcam process thread.
def process_webcam_image(image_path):
    image_path = cv2.cvtColor(image_path,cv2.COLOR_BGR2RGB)
    processedImage = show_inference(detection_model, image_path, False, True, False, 0)
    processedImage = cv2.cvtColor(processedImage,cv2.COLOR_RGB2BGR)
    return processedImage
    #show_inference(masking_model, image_path, True)

#This handler will handle the webcam feed.
def webcam_handler(mirror=False):
    vs = VideoStream().start() #Will open the first webcam it finds. Probably the one hooked to USB if you don't have an integrated one.
    while True:
        key = cv2.waitKey(1) & 0xFF

        frame = vs.read()
        frame = im.resize(frame, height=240)
        frame = im.resize(frame, width=320)
        #if mirror:
        #    img = cv2.flip(img,1)

        processedImage = process_webcam_image(frame)
        
        cv2.imshow("Object Detection Webcam Stream", processedImage)

        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()
    vs.stop()

#One of these below needs to be uncommented...
webcam_handler()
