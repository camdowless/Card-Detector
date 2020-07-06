from glob import glob
import random

import matplotlib.image as mpimg
from hickle import hickle
import PIL
import matplotlib.pyplot as plt

dtd_dir="dtd/images/"
backgrounds_hkl_fn = "backgrounds.hkl"
bg_images=[]
for subdir in glob(dtd_dir+"/*"):
    for f in glob(subdir+"/*.jpg"):
        bg_images.append(mpimg.imread(f))
print("Nb of images loaded :",len(bg_images))
print("Saved in :",backgrounds_hkl_fn)
hickle.dump(bg_images,open(backgrounds_hkl_fn, mode='w'))


class Backgrounds():
    def __init__(self, backgrounds_hkl_fn=backgrounds_hkl_fn):
        self._images = hickle.load(backgrounds_hkl_fn)
        self._nb_images = len(self._images)
        print("Nb of images loaded :", self._nb_images)

    def get_random(self, display=False):
        bg = self._images[random.randint(0, self._nb_images - 1)]
        if display: plt.imshow(bg)
        return bg


backgrounds = Backgrounds()
_=backgrounds.get_random(display=True)