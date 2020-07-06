import cv2
import crop as c
import os

"""
    Resizes original card photos to be cropped
"""

"""
for image in os.listdir("old_cards"):
    im = cv2.imread(f"old_cards/{image}")
    new = cv2.resize(im, (0, 0), fy = .39, fx = .39)
    new = c.extract_card(new)
    cv2.imwrite(f"cards/{image}", new)
"""



"""
    Resizes training data to fit YOLO model
"""

data = "valid"
dir = os.path.join("yolo_training_data", data, "images")
for image in os.listdir(dir):
    im = cv2.imread(f"{dir}/{image}")
    new = cv2.resize(im, (416, 416))
    cv2.imwrite(os.path.join(dir, image), new)