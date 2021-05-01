import numpy as np
from PIL import Image, ImageFilter

from vine import Vine


class Sigmoid:
    """ Sigmoid function which was modified to cross (0, 0) and (1, 1) 
    
    Based on https://hackernoon.com/ease-in-out-the-sigmoid-factory-c5116d8abce9
    See the functions here: https://www.desmos.com/calculator/pa0cjsv66m
    """
    def __init__(self, k=4):
        self.k = k

    def _s(self, x):
        return 1 / (1 + np.exp(-self.k * x)) - 0.5

    def ease(self, x):
        return 0.5 * (self._s(2*x-1) / self._s(1) + 1)


def main():
    size = (1000, 1000)
    center = (size[0] // 2, size[1] // 2)

    images = []
    frame_count = 160
    for frame in range(frame_count):
        print(f"Drawing frame #{frame+1}/{frame_count}!")
        img = Image.new("RGB", size, (0, 0, 0))
        vine_count = 5
        build_phase = frame if frame < frame_count//2 else frame_count - frame
        vines = [
            Vine(center, 190, 15, rotate_degrees=frame+i*(360/vine_count), build_phase=.05*build_phase)
            for i in range(vine_count)
        ]
        for vine in vines:
            vine.draw_on_image(img)
        img = img.filter(ImageFilter.GaussianBlur(5))
        images.append(img)
    images[0].save("growing_star.gif", save_all=True, append_images=images[1:], loop=0)


if __name__ == "__main__":
    main()
