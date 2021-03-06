from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from color_gradient import ColorGradient
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

    def __call__(self, x):
        return 0.5 * (self._s(2 * x - 1) / self._s(1) + 1)

    def smooth(self, val, min_val, max_val):
        return val * self((val - min_val) / max_val)


def main():
    size = (1000, 1000)
    center = (size[0] // 2, size[1] // 2)

    frame_count = 160
    vine_count = 5
    angle_between_vines = 360 / vine_count
    target_angle = angle_between_vines * 2
    angle_step_size = target_angle / frame_count
    sigmoid = Sigmoid(2)
    iterations = [
        (0, 1), (-90, 0), (-72, 1), (90, 1), (-72, 0), (90, 0),
        (-120, 0), (45, 1), (-180, 0), (-180, 1), (-45, 0), (360 / 7, 1)
    ]
    img_dir = Path('images')
    img_dir.mkdir(exist_ok=True)
    for iteration, (add_degrees, axis_to_invert) in enumerate(iterations):
        for frame in range(frame_count):
            overall_frame_index = frame + frame_count * iteration
            print(f"Drawing frame #{overall_frame_index}/{frame_count * len(iterations)}!")
            img = Image.new("RGB", size, (0, 0, 0))
            build_phase = frame if frame < frame_count // 2 else frame_count - frame
            build_phase = sigmoid.smooth(build_phase, 0, frame_count // 2)
            # angle = sigmoid.smooth(angle_step_size * frame, 0, target_angle)
            angle = angle_step_size * frame
            vines = []
            for v in range(vine_count):
                vines.append(Vine(
                    center, 190, 15, rotate_degrees=angle + v * angle_between_vines,
                    build_phase=.05 * build_phase, add_degrees_to_angle=add_degrees,
                    axis_to_invert=axis_to_invert,
                    color=ColorGradient(("#51007D", 1), ("#FFD600", 1.5)),
                ))
            for vine in vines:
                vine.draw_on_image(img)
            img = img.filter(ImageFilter.GaussianBlur(5))
            img.save(img_dir / f"{overall_frame_index}.png")
    # images[0].save("growing_star.gif", save_all=True, append_images=images[1:], loop=0, duration=30)


if __name__ == "__main__":
    main()
