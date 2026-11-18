from typing import List, Iterator

from geomant.geom import Vector


class Tile:
    def __init__(self):
        pass


class Structure:
    def __init__(self, *children: 'Structure'):
        self.children: List[Structure] = []
        self.tiles: List[Tile] = []
        for child in children:
            if isinstance(child, Structure):
                self.children.append(child)
            elif isinstance(child, Tile):
                self.tiles.append(child)
            else:
                raise TypeError(type(child))

    def __iter__(self) -> Iterator[Tile]:
        for child in self.children:
            yield from child
        yield from self.tiles


def render_structure(structure: Structure, *, box: bool = True, zero: bool = True, fill: str = '░') -> str:
    """

        >>> from geomant.geom import A, B, C, D, R, R3

        >>> sq = Square()
        >>> tri = Triangle()
        >>> quarter = Structure(
        ...     sq,
        ...     tri * R + A,
        ...     tri * R3 + A + B,
        ... )
        >>> print(render_structure(quarter))
        ##############
        #         -x #
        #   x---x-░| #
        #   |░░░|░-x #
        #   0---x-   #
        ##############
        >>> f = R3 + (A * 2 + B)
        >>> print(render_structure(Structure(
        ...     quarter,
        ...     quarter * f,
        ...     quarter * f ** 2,
        ...     quarter * f ** 3,
        ...     sq * R + A + D,
        ... )))
        ##################
        #       -x---x   #
        # x---x-░|░░░|   #
        # |░░░|░-x---x   #
        # x---x-░░\░/░\  #
        #  \░/░\░░-x---x #
        #   x---x-░|░░░| #
        #   |░░░|░-x---x #
        #   0---x-       #
        ##################

    """
    vecs = tuple((vec, char) if not isinstance(vec, tuple) else vec
        for vec in vecs)
    def to_coord(v: Vector) -> Tuple[int, int]:
        return 2 * (v.x0 + v.x1) + v.x2, -(v.x1 + 2 * (v.x2 + v.x3))
    coords = tuple(map(to_coord, (vec for vec, char in vecs)))
    min_x = min(x for x, y in coords)
    min_y = min(y for x, y in coords)
    max_x = max(x for x, y in coords)
    max_y = max(y for x, y in coords)
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    pixels = [[' ' for x in range(w)] for y in range(h)]
    for (vec, char), (x, y) in zip(vecs, coords):
        pixels[y - min_y][x - min_x] = char
    lines = [''.join(line) for line in pixels]
    if box:
        lines = ['|' + line + '|' for line in lines]
        horiz = '+' + '-' * w + '+'
        lines.insert(0, horiz)
        lines.append(horiz)
    return '\n'.join(lines)
