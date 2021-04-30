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

    invert = np.array([
        [0, -1],
        [1, 0],
    ])

    def __init__(self, start_pos: tuple[float, float], length: float, thickness: float, rotate_degrees=0.0, flip=False):
        rotate_radians = rotate_degrees / 180 * np.pi
        rotation_matrix = np.array([
            [np.cos(rotate_radians), -np.sin(rotate_radians)],
            [np.sin(rotate_radians), np.cos(rotate_radians)],
        ])
        self.flipped = flip
        shape = np.copy(Vine.shape.T)
        if flip:
            shape[:, 0] *= -1
        rotated = shape @ rotation_matrix
        nodes = rotated.T * length + np.array(start_pos)[:, None]
        self.thickness = thickness
        self.length = length
        self.curve = bezier.Curve(nodes, degree=len(nodes[0]) - 1)

    def get_tangent_at(self, at: float):
        return normalize(self.curve.evaluate_hodograph(at), axis=0)

    def get_angle_at(self, at: float):
        x, y = self.get_tangent_at(at).T[0]
        # Subtract 90 degrees to account that the unit circle 0 is to the right,
        # whereas the vine defined in Vine.shape points down.
        return np.arctan2(y, x) - np.pi/2

    def get_normal_at(self, at: float):
        return Vine.invert @ self.get_tangent_at(at)

    def draw_on_image(self, image: Image):
        draw = ImageDraw.Draw(image)
        dots = np.arange(0, 1, 1 / self.curve.length)
        thickness = thinning(dots)
        evaluated = self.curve.evaluate_multi(dots)
        for t, (d, (x, y)) in zip(thickness, zip(dots, zip(*evaluated))):
            tangent_line = self.get_normal_at(d) * self.thickness * t
            delta_x, delta_y = tangent_line.T[0]
            draw.line((x - delta_x, y - delta_y, x + delta_x, y + delta_y), fill=255, width=5)
        # nx, ny = self.get_tangent_at(.1) * 40
        # print(nx, ny)
        # px, py = self.curve.evaluate(.1).T[0]
        # draw.line((px, py, px+nx, py+ny), fill=(255,255,255), width=5)

    def get_child_vine(self, at: float):
        start_pos = tuple(self.curve.evaluate(at).flatten())
        angle = -self.get_angle_at(at) * 180 / np.pi  # + np.random.uniform(-30, 30)
        print(angle)
        thinned = thinning(np.array([at]))
        length = self.length * thinned[0]
        thickness = thinned * self.thickness
        return Vine(start_pos, length, thickness, rotate_degrees=angle, flip=not self.flipped)


def main():
    img = Image.new("RGB", size, (0, 0, 0))
    vine_count = 5
    vines = [
        Vine(center, 200, 15, rotate_degrees=45+i*360/vine_count)
        for i in range(vine_count)
    ]
    unused = set(vines)
    print(unused)
    for index in range(100):
        parent_vine = np.random.choice(list(unused))
        unused.remove(parent_vine)
        # parent_vine = vines[-1]
        # parent_vine = vines[int(np.random.beta(1, 2) * len(vines))]
        at = .1
        # at = np.random.uniform()
        vines.append(parent_vine.get_child_vine(at))
        unused.add(vines[-1])
    for vine in vines:
        vine.draw_on_image(img)
    img.save("vines.png")


if __name__ == "__main__":
    main()
