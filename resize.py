import cv2
import crop as c
import os

for image in os.listdir("old_cards"):
    im = cv2.imread(f"old_cards/{image}")
    new = cv2.resize(im, (0, 0), fy = .39, fx = .39)
    new = c.extract_card(new)
    cv2.imwrite(f"cards/{image}", new)

"""
count = 0
for image in os.listdir("old_cards"):
    im = cv2.imread(f"old_cards/{image}")
    im = cv2.resize(im, (0, 0), fy = .39, fx = .39)
    if c.extract_card(im) is False:
        count += 1
        print(f"{image} is fucked")
print(f"{count} fucked images")"""

'''im = cv2.imread("old_cards/Jd.JPG")
new = cv2.resize(im, (0,0), fy = .39, fx = .39)
c.extract_card(new)'''