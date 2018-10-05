
# coding: utf-8

# ## Face and Facial Keypoint detection
# 
#  The neural network expects a Tensor of a certain size as input and, so, to detect any face, you'll first have to do some pre-processing.
# 
# 1. Detect all the faces in an image using a face detector (we'll be using a Haar Cascade detector in this notebook).
# 2. Pre-process those face images so that they are grayscale, and transformed to a Tensor of the input size that your net expects. This step will be similar to the `data_transform` you created and applied in Notebook 2, whose job was tp rescale, normalize, and turn any iimage into a Tensor to be accepted as input to your CNN.
# 3. Use your trained model to detect facial keypoints on the image.
# 
# ---

# In the next python cell we load in required libraries for this section of the project.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Select an image 
# 
# Select an image to perform facial keypoint detection on; you can select any image of faces in the `images/` directory.

# In[2]:


import cv2
# load in color image for face detection
image = cv2.imread('images/obamas.jpg')

# switch red and blue color channels 
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
fig = plt.figure(figsize=(9,9))
plt.imshow(image)


# ## Detect all faces in an image
# 
# In the code below, we loop over each face in the original image and draw a red square on each face (in a copy of the original image, so as not to modify the original). You can even [add eye detections](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html) as an *optional* exercise in using Haar detectors.
# 
# An example of face detection on a variety of images is shown below.
# 
# <img src='images/haar_cascade_ex.png' width=80% height=80%/>
# 

# In[3]:


# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()
print(faces)
# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

fig = plt.figure(figsize=(9,9))

plt.imshow(image_with_detections)


# ## Loading in a trained model
# 
# Once you have an image to work with (and, again, you can select any image of faces in the `images/` directory), the next step is to pre-process that image and feed it into your CNN facial keypoint detector.
# 
# First, load your best model by its filename.

# In[4]:


import torch
from models import Net

net = Net()

## TODO: load the best saved model parameters (by your path name)
## You'll need to un-comment the line below and add the correct name for *your* saved model
net.load_state_dict(torch.load('saved_models/keypoints_model_Best.pt'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval()


# ## Keypoint detection
# 
# Now, we'll loop over each detected face in an image (again!) only this time, you'll transform those faces in Tensors that your CNN can accept as input images.
# 
# ###  Transform each detected face into an input Tensor
# 
# You'll need to perform the following steps for each detected face:
# 1. Convert the face from RGB to grayscale
# 2. Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
# 3. Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
# 4. Reshape the numpy image into a torch image.
# 
# **Note**: The sizes of faces detected by a Haar detector and the faces your network has been trained on are of different sizes. If you find that your model is generating keypoints that are too small for a given face, try adding some padding to the detected `roi` before giving it as input to your model.
# 
# You may find it useful to consult to transformation code in `data_load.py` to help you perform these processing steps.
# 
# 
# ### Detect and display the predicted keypoints
# 
# After each face has been appropriately converted into an input Tensor for your network to see as input, you can apply your `net` to each face. The ouput should be the predicted the facial keypoints. These keypoints will need to be "un-normalized" for display, and you may find it helpful to write a helper function like `show_keypoints`. You should end up with an image like the following with facial keypoints that closely match the facial features on each individual face:
# 
# <img src='images/michelle_detected.png' width=30% height=30%/>
# 
# 
# 

# In[15]:


from torchvision import transforms
from data_load import Rescale, RandomCrop, Normalize, ToTensor
from data_load import FacialKeypointsDataset
import torch.nn as nn
import torch.nn.functional as F
image_copy = np.copy(image)

num=0
print(faces)
# loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:
    num = num+1
    # Select the region of interest that is the face in the image 
    roi = image_copy[y:y+h, x:x+w]
    ## Convert the face region from RGB to grayscale
    roi=cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    #plt.imshow(roi, cmap='gray')
    ## Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi=roi/255.0
    print(x,w,y,h)
    ## Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    ## Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    ##  Make facial keypoint predictions using your loaded, trained network
    if h > w:
                new_h, new_w = 224 * h / w, 224
    else:
                new_h, new_w = 224, 224 * w / h
    
    new_h, new_w = int(new_h), int(new_w)
    roi = cv2.resize(roi, (new_w, new_h))
    roi= roi[np.newaxis,np.newaxis,...]
    roi=torch.from_numpy(roi)
    roi=roi.type(torch.FloatTensor)
    pad = nn.ConstantPad2d((1,1), 0) #image padding with zero
    roi = pad(roi)
    output=net(roi)
    show_all_keypoints(roi, output)
  ##O: Display each detected face and the corresponding keypoints 
    


# In[6]:


def show_all_keypoints(image, predicted_key_pts):
    """Show image with predicted keypoints"""
    # image is grayscale
    image = image.data   # get the image from it's Variable wrapper
    image = image.numpy()   # convert to numpy array from a Tensor
    image = np.squeeze(image)
    predicted_key_pts = predicted_key_pts.data
    predicted_key_pts = predicted_key_pts.numpy()
    predicted_key_pts = predicted_key_pts*50.0+100
    predicted_key_pts=np.resize(predicted_key_pts, (68,2))
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    plt.imshow(image, cmap='gray')
    plt.show()
   

