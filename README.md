# Card Detector - A computer vision detection model trained on playing cards.
## Written by Cam Dowless

### Tools used:
  - Python
  - Google Colab notebooks for the GPU
  - [YOLOv4 object detection algorithm](https://arxiv.org/abs/2004.10934)
  - [Darknet framework](https://pjreddie.com/darknet/)
  - [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

` Data Generation.ipynb`:
  - Generates annotated training data for the machine learning model. I started by taking photos of each playing card (`/cards_og`), cropping the background out (`/cards_cropped`), and making 50 copies increasing & decreasing the brightness levels. 
  - Finds suit/value indicators in the top left, bottom right corners and draws a rectangle around them, saving the rectangle coordinates. 
  - Takes 2 cards and an image from the DTD dataset, and places the cards in a random position. Generates an XML file containing the coordinates of the bouding boxes, the card classifications, and a unique image name. 
  - For 3 cards, they are placed in a "fanned out" position, and the same process ensues. 
  
`convert_voc_yolo.py`: 
  - Converts the XML annotation for each playing card into a format readable by the YOLOv4 training algorithm. 
 
 `Card_Detection.ipynb`: 
  - Installs the proper frameworks & starts training.
  - Training time takes ~24 hours, even with Google's Tesla P100 GPU. 
  - You can then evaluate the model on an image or video. 
  
   <img src="https://github.com/camdowless/Card-Detector/blob/master/p6.jpg?" width="400" height="525">
   
