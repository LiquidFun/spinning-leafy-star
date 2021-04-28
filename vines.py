import bezier
import numpy as np
from PIL import Image, ImageDraw
from sklearn.preprocessing import normalize

size = (1000, 1000)
center = (size[0]//2, size[1]//2)


def epow(x):
    f = .5
    return np.e ** (-f * x) - np.e ** (-f)


def sigmoid(x):
    return 1 - (1 / (1 + np.e ** (2 - 6 * x)))


def thinning(arr: np.ndarray):
    return np.clip(0, 1, np.apply_along_axis(sigmoid, axis=0, arr=arr))


class Vine:
    shape = np.array([
        [0.0, 0.0, 0.9, 1.0, 0.6],
        [0.0, 1.5, 1.5, 0.6, 0.5],
    ])

    def __init__(self, start_pos: tuple[float, float], length: float, thickness: float, rotate_degrees=0.0):
        rotate_radians = rotate_degrees / 180 * np.pi
        rotation_matrix = np.array([
            [np.cos(rotate_radians), -np.sin(rotate_radians)],
            [np.sin(rotate_radians), np.cos(rotate_radians)],
        ])
        rotated = Vine.shape.T @ rotation_matrix
        nodes = rotated.T * length + np.array(start_pos)[:, None]
        self.thickness = thickness
        self.length = length
        self.curve = bezier.Curve(nodes, degree=len(nodes[0]) - 1)

    def draw_on_image(self, image: Image):
        draw = ImageDraw.Draw(image)
        dots = np.arange(0, 1, 1 / self.curve.length)
        thickness = thinning(dots)
        evaluated = self.curve.evaluate_multi(dots)
        for t, (d, (x, y)) in zip(thickness, zip(dots, zip(*evaluated))):
            tangent_line = normalize(self.curve.evaluate_hodograph(d), axis=0) * 20 * t
            delta_x, delta_y = tangent_line[1][0], -tangent_line[0][0]
            draw.line((x - delta_x, y - delta_y, x + delta_x, y + delta_y), fill=255, width=5)


def main():
    img = Image.new("RGB", size, (0, 0, 0))
    vines = [
        Vine(center, 300, 20),
        Vine(center, 300, 20, rotate_degrees=180),
    ]
    # for index in range(10):
    #     chosen = np.random.choice(vines)
    #     print(chosen)
    for vine in vines:
        vine.draw_on_image(img)
        at = np.random.rand(0, 1)
    img.save("vines.png")


if __name__ == "__main__":
    main()
