import hickle
import cv2
import os
from glob import glob
import hull as h
import random
import data


card_suits=['s','h','d','c']
card_values=['A','K','Q','J','10','9','8','7','6','5','4','3','2']



class Cards():
    """
        Loads the 'cards.hkl' file into an instance of Card class
    """
    def __init__(self):
        self._cards = hickle.load("cards.hkl")
        # self.cards is a dictionary
        # Keys: card names, Kc, 10s, 7d
        # Values: lists of (img, hullHL, hullLR)

        self._num_cards_by_value = {k : len(self._cards[k]) for k in self._cards}
        print("No. of cards loaded per name: ", self._num_cards_by_value)

    def get_random(self, card_name = None, display=False):
        if card_name is None:
            card_name = random.choice(list(self._cards.keys()))
        card, hull1, hull2 = self._cards[card_name][random.randint(0, self._num_cards_by_value[card_name]-1)]
        if display:
            h.display_img(card, [hull1, hull2], "rgb")
        return card,card_name,hull1,hull2


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

def edit_cards():
    images = list()

    for image_string in os.listdir("cropped_cards"):
        image = cv2.imread(f"cropped_cards/{image_string}")
        for i in range(0, 100, 4):
            image_string = image_string.replace(".JPG", "")
            path = os.path.join("edited_cards", image_string)
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
            bright_image = change_brightness(image, i)
            dark_image = change_brightness(image, i * -1)
            bright_path = os.path.join(path, f"{image_string}_{i}_b.JPG")
            dark_path = os.path.join(path, f"{image_string}_{i}_d.JPG")
            cv2.imwrite(bright_path, bright_image)
            cv2.imwrite(dark_path, dark_image)


def saveToHickle():
    imgs_dir = "edited_cards"
    cards = {}
    for suit in card_suits:
        for value in card_values:
            card_name = value+suit
            card_dir = os.path.join(imgs_dir, card_name)
            if not os.path.isdir(card_dir):
                print(f"{card_dir} does not exist")
                continue
            cards[card_name] = []
            for f in glob(card_dir + "/*.JPG"):
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                hullHL = h.findHull(img, data.refCornerHL, debug="no")
                if hullHL is None:
                    print(f"File {f} not used.")
                    continue
                hullLR = h.findHull(img, data.refCornerLR, debug="no")
                if hullLR is None:
                    print(f"File {f} not used.")
                    continue
                # Store the image in RBG format
                # OpenCV is no longer needed
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                cards[card_name].append((img, hullHL, hullLR))
            print(f"No. images for {card_name} : {len(cards[card_name])}")
    print("Saved in: ", "cards.hkl")
    hickle.dump(cards, "cards.hkl", mode='w')


xml_body_1="""<annotation>
        <folder>FOLDER</folder>
        <filename>{FILENAME}</filename>
        <path>{PATH}</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>{WIDTH}</width>
                <height>{HEIGHT}</height>
                <depth>3</depth>
        </size>
"""
xml_object=""" <object>
                <name>{CLASS}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>{XMIN}</xmin>
                        <ymin>{YMIN}</ymin>
                        <xmax>{XMAX}</xmax>
                        <ymax>{YMAX}</ymax>
                </bndbox>
        </object>
"""
xml_body_2="""</annotation>        
"""

def create_voc_xml(xml_file, img_file,listbba,display=False):

    with open(xml_file,"w") as f:
        f.write(xml_body_1.format(**{'FILENAME':os.path.basename(img_file), 'PATH':img_file,'WIDTH':data.imgW,'HEIGHT':data.imgH}))
        for bba in listbba:
            f.write(xml_object.format(**{'CLASS':bba.classname,'XMIN':bba.x1,'YMIN':bba.y1,'XMAX':bba.x2,'YMAX':bba.y2}))
        f.write(xml_body_2)
        if display: print("New xml",xml_file)


cards = Cards()
_ = cards.get_random(display=True)