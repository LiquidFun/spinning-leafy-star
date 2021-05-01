from colour import Color


class ColorGradient:
    def __init__(self, from_color_at_phase: tuple[str, float], to_color_at_phase: tuple[str, float]):
        self.from_phase = from_color_at_phase[1]
        self.to_phase = to_color_at_phase[1]
        from_color = Color(from_color_at_phase[0])
        to_color = Color(to_color_at_phase[0])

        def to_255_scale(color: Color):
            return tuple(map(lambda c: round(c * 255), color.rgb))

        self.color_range = list(map(to_255_scale, from_color.range_to(to_color, 100)))

    def interpolate(self, at):
        at_index = round((at - self.from_phase) / self.to_phase * (len(self.color_range)-1))
        return self.color_range[max(0, min(at_index, len(self.color_range)-1))]
