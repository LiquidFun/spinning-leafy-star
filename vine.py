from typing import Annotated, Literal

import bezier
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.preprocessing import normalize


class Vine:
    """ Vine class which uses bezier curves to draw itself on an image

    """

    def __init__(
            self,
            start_pos: tuple[float, float],
            length: float,
            thickness: float,
            *,
            color: tuple[int, int, int] = (255, 255, 255),
            build_phase=1e9,
            rotate_degrees=0.0,
            flip=False,
            depth=1,
            max_child_vines=16,
            grow_child_at=0.1,
            axis_to_invert: int = 1,
            add_degrees_to_angle: float = 0,
    ):
        self.vine_shape = np.array([
            [0.0, 1.5, 1.5, 0.6, 0.5],
            [0.0, 0.0, 0.9, 1.0, 0.6],
        ])
        self.axis_to_invert = axis_to_invert
        self.add_degrees_to_angle = add_degrees_to_angle
        self.grow_child_at = grow_child_at
        self.color = color
        self.flipped = flip
        self.depth = depth
        self.max_child_vines = max_child_vines
        self.build_phase = build_phase
        self.thickness = thickness
        self.length = length
        nodes = self._get_nodes_for_curve(start_pos, rotate_degrees)
        self.curve = bezier.Curve(nodes, degree=len(nodes[0]) - 1)
        self.child_vine = self._create_child_vine(grow_child_at) if depth < max_child_vines else None

    def _get_nodes_for_curve(self, start_pos, rotate_degrees):
        rotate_radians = rotate_degrees / 180 * np.pi
        rotation_matrix = np.array([
            [np.cos(rotate_radians), -np.sin(rotate_radians)],
            [np.sin(rotate_radians), np.cos(rotate_radians)],
        ])
        shape = np.copy(self.vine_shape.T)
        if self.flipped:
            shape[:, self.axis_to_invert] *= -1
        rotated = shape @ rotation_matrix
        return rotated.T * self.length + np.array(start_pos)[:, None]

    def _create_child_vine(self, at: float):
        start_pos = tuple(self.curve.evaluate(at).flatten())
        angle = -self.get_angle_at(at) * 180 / np.pi  # + np.random.uniform(-30, 30)
        thinned = self._thinning(np.array([at]))
        length = self.length * thinned[0]
        thickness = thinned * self.thickness
        return Vine(start_pos, length, thickness, rotate_degrees=angle, flip=not self.flipped,
                    depth=self.depth + 1, max_child_vines=self.max_child_vines, build_phase=self.build_phase - .2,
                    color=self.color)

    @staticmethod
    def _sigmoid(x: float):
        """ Modified sigmoid function which goes from y=1 to 0 within x=0...1
        See here: https://www.desmos.com/calculator/s5lr4vi48n
        """
        return 1 - (1 / (1 + np.e ** (2 - 6 * x)))

    def _thinning(self, arr: np.ndarray):
        """ Apply inverse sigmoid function to every element in arr """
        return np.clip(0, 1, np.apply_along_axis(self._sigmoid, axis=0, arr=arr))

    def get_tangent_at(self, at: float) -> np.ndarray:
        return normalize(self.curve.evaluate_hodograph(at), axis=0)

    def get_normal_at(self, at: float) -> np.ndarray:
        return np.array([[0, -1], [1, 0]]) @ self.get_tangent_at(at)

    def get_angle_at(self, at: float) -> float:
        """ Returns angle in radians of the tangent for 0.0 <= at <= 1.0 """
        x, y = self.get_tangent_at(at).T[0]
        return np.arctan2(y, x) + self.add_degrees_to_angle

    def draw_on_image(self, image: Image):
        draw = ImageDraw.Draw(image)
        phase = min(1.0, self.build_phase)
        dots = np.arange(0, phase, 1 / self.curve.length)
        thickness = self._thinning(dots)
        evaluated = self.curve.evaluate_multi(dots)
        for t, (d, (x, y)) in zip(thickness, zip(dots, zip(*evaluated))):
            tangent_line = self.get_normal_at(d) * self.thickness * t * np.clip(0, 1, self.build_phase)
            delta_x, delta_y = tangent_line.T[0]
            draw.line((x - delta_x, y - delta_y, x + delta_x, y + delta_y), fill=self.color, width=5)
        if self.child_vine:
            self.child_vine.draw_on_image(image)
        draw_tangent = False
        if draw_tangent:
            nx, ny = self.get_tangent_at(self.grow_child_at) * 40
            px, py = self.curve.evaluate(self.grow_child_at).T[0]
            draw.line((px, py, px + nx, py + ny), fill=(255, 255, 255), width=5)


def main():
    size = (1000, 1000)
    center = (size[0] // 2, size[1] // 2)

    img = Image.new("RGB", size, (0, 0, 0))
    vine_count = 5
    vines = [
        Vine(center, 190, 15, rotate_degrees=i * (360 / vine_count),
             build_phase=.7, color=(255, 55, 0))
        for i in range(vine_count)
    ]
    for vine in vines:
        vine.draw_on_image(img)
    img = img.filter(ImageFilter.GaussianBlur(5))
    img.save("vines.png")


if __name__ == "__main__":
    main()
