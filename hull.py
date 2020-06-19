import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from glob import glob
import numpy as np
import data


def findHull(img, corner = data.refCornerHL, debug = False):
    """
            Find in the zone 'corner' of image 'img' and return, the convex hull delimiting
            the value and suit symbols
            'corner' (shape (4,2)) is an array of 4 points delimiting a rectangular zone,
            takes one of the 2 possible values : refCornerHL or refCornerLR
            debug=
        """

    kernel = np.ones((3,3), np.uint8)
    corner = corner.astype(np.int)

    x1 = int(corner[0][0])
    y1 = int(corner[0][1])
    x2 = int(corner[2][0])
    y2 = int(corner[2][1])
    w = x2 - x1
    h = y2 - y1
    zone = img[y1:y2, x1:x2].copy()

    strange_cnt = np.zeros_like(zone)
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    thld = cv2.Canny(gray, 30, 200)
    thld = cv2.dilate(thld, kernel, iterations=1)
    if debug != "no":
        cv2.imshow("thld", thld)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Find contours
    contours, _ = cv2.findContours(thld.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 30 # Reject contours with a small area. TWEAK 'zoom' dependent
    min_solidity = 0.3 # reject contours with a low solidity TWEAK

    concat_contour = None
    ok = True
    for c in contours:
        area = cv2.contourArea(c)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        # Determine the center of gravity (cx,cy) of the contour
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        #  abs(w/2-cx)<w*0.3 and abs(h/2-cy)<h*0.4 : TWEAK, the idea here is to keep only the contours which are closed to the center of the zone
        if area >= min_area and abs(w / 2 - cx) < w * 0.3 and abs(h / 2 - cy) < h * 0.4 and solidity > min_solidity:
            if debug != "no":
                cv2.drawContours(zone, [c], 0, (255, 0, 0), -1)
            if concat_contour is None:
                concat_contour = c
            else:
                concat_contour = np.concatenate((concat_contour, c))
        if debug != "no" and solidity <= min_solidity:
            print("Solidity", solidity)
            cv2.drawContours(strange_cnt, [c], 0, 255, 2)
            cv2.imshow("Strange contours", strange_cnt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if concat_contour is not None:
        # At this point, we suppose that 'concat_contour' contains only the contours corresponding the value and suit symbols
        # We can now determine the hull
        hull = cv2.convexHull(concat_contour)
        hull_area = cv2.contourArea(hull)
        # If the area of the hull is to small or too big, there may be a problem
        min_hull_area = 940  # TWEAK, deck and 'zoom' dependant
        max_hull_area = 2120  # TWEAK, deck and 'zoom' dependant
        if hull_area < min_hull_area or hull_area > max_hull_area:
            ok = False
            if debug != "no":
                print("Hull area=", hull_area, "too large or too small")
        # So far, the coordinates of the hull are relative to 'zone'
        # We need the coordinates relative to the image -> 'hull_in_img'
        hull_in_img = hull + corner[0]

    else:
        ok = False

    if debug != "no":
        if concat_contour is not None:
            cv2.drawContours(zone, [hull], 0, (0, 255, 0), 1)
            cv2.drawContours(img, [hull_in_img], 0, (0, 255, 0), 1)
        cv2.imshow("Zone", zone)
        cv2.imshow("Image", img)
        if ok and debug != "pause_always":
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(0)
        if key == 27:
            return None
    if ok == False:
        return None

    return hull_in_img


def display_img(img, polygons=None, channels="bgr", size=9):
    """
        Function to display an inline image, and draw optional polygons (bounding boxes, convex hulls) on it.
        Use the param 'channels' to specify the order of the channels ("bgr" for an image coming from OpenCV world)
    """
    if polygons is None:
        polygons = []
    if not isinstance(polygons,list):
        polygons=[polygons]
    if channels=="bgr": # bgr (cv2 image)
        nb_channels=img.shape[2]
        if nb_channels==4:
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGBA)
        else:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig,ax=plt.subplots(figsize=(size,size))
    ax.set_facecolor((0,0,0))
    ax.imshow(img)


    for polygon in polygons:
        # An polygon has either shape (n,2),
        # either (n,1,2) if it is a cv2 contour (like convex hull).
        # In the latter case, reshape in (n,2)
        if len(polygon.shape)==3:
            polygon=polygon.reshape(-1,2)
        patch=patches.Polygon(polygon,linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(patch)



def testDimensions():

    """
        These 5 lines take 5 random cards and draw a box around the corners.
        Used to test the dimensions CardW, CardH, CornerXMin, CornerXMax, CornerYMin, and CornerYMax
    """
    imgs_dir = "edited_cards"
    imgs_fns = glob(imgs_dir + "/*.JPG")
    for _ in range(0, 10):
        img_fn = random.choice(imgs_fns)
        display_img(cv2.imread(img_fn, cv2.IMREAD_UNCHANGED), polygons=[data.refCornerHL, data.refCornerLR])


def testHull():
    debug="no"
    imgs_dir="edited_cards"
    imgs_fns=glob(imgs_dir+"/*.JPG")
    img_fn=random.choice(imgs_fns)
    print(img_fn)
    img=cv2.imread(img_fn,cv2.IMREAD_UNCHANGED)

    hullHL=findHull(img,data.refCornerHL,debug=debug)
    hullLR=findHull(img,data.refCornerLR,debug=debug)
    display_img(img,[data.refCornerHL,data.refCornerLR,hullHL,hullLR])


