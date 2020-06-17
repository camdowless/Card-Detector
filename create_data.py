import cv2
import os

def change_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final = cv2.merge((h, s, v))
    else:
        value = abs(value)
        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value
        final = cv2.merge((h, s, v))
    img = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
    return img

images = list()

for image_string in os.listdir("cropped_cards"):
    image = cv2.imread(f"cropped_cards/{image_string}")
    for i in range(0, 100, 4):
        bright_image = change_brightness(image, i)
        dark_image = change_brightness(image, i*-1)
        cv2.imwrite(f"edited_cards/{image_string}_{i}_b.JPG", bright_image)
        cv2.imwrite(f"edited_cards/{image_string}_{i}_d.JPG", dark_image)
