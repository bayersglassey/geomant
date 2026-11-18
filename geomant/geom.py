import operator
from typing import Tuple, Iterable, Union
from itertools import repeat


VectorTuple = Tuple[int, int, int, int]


def _multiply(xx: VectorTuple, yy: VectorTuple) -> VectorTuple:
    """Encodes a matrix for multiplying two Vector instances"""
    x0, x1, x2, x3 = xx
    y0, y1, y2, y3 = yy

    """
    NOTE: I derived the following expressions somewhat hackily using sympy:

        import sympy

        A = 1
        B = sympy.symbols('B')
        C = B*B
        D = B*B*B
        a1, b1, c1, d1 = sympy.symbols('a1 b1 c1 d1')
        a2, b2, c2, d2 = sympy.symbols('a2 b2 c2 d2')

        mapping={'B**2': 'C', 'B**3': 'D', 'B**4': '(C-A)', 'B**5': '(D-B)', 'B**6': '(-A)'}
        def vec_expand(expr):
            s = str(expr)
            for k, v in mapping.items():
                s = s.replace(k, v)
            return s

        expr = (a1*A + b1*B + c1*C + d1*D) * (a2*A + b2*B + c2*C + d2*D)

        sympy.expand(expr)
        expr = eval(vec_expand(expr))

    """
    return (
        x0*y0 - x1*y3 - y1*x3 - x2*y2 - x3*y3,
        x0*y1 + y0*x1 - x2*y3 - y2*x3,
        x0*y2 + y0*x2 + x1*y1 + x1*y3 + y1*x3 + x2*y2,
        x0*y3 + y0*x3 + x1*y2 + y1*x2 + x2*y3 + y2*x3,
    )


class Vector:
    """A vector in a 4-dimensional vector space.

        Here are the 4 dimensions of this vector space:

           x3 x2
           | / _x1
           |/_/
           0 --- x0

        They correspond to these unit vectors:

           D  C
           | / _B
           |/_/
           0 --- A

        There is an operator, R, which applies a 30-degree counter-clockwise
        rotation, e.g. RA = B, RB = C, etc.

        When vectors are multiplied (using Python's "matrix multiplication"
        operator, i.e. "@"), the unit vector B acts like R, C like R**2, etc.

        >>> v = Vector(1, 2, 0, 0)
        >>> -v
        Vector(-1, -2, 0, 0)
        >>> v * 3
        Vector(3, 6, 0, 0)
        >>> v // 2
        Vector(0, 1, 0, 0)

        >>> w = Vector(0, 3, 4, 0)
        >>> v + w
        Vector(1, 5, 4, 0)
        >>> v - w
        Vector(1, -1, -4, 0)

        >>> v2 = v.copy()
        >>> v2
        Vector(1, 2, 0, 0)
        >>> v2 += w
        >>> v2
        Vector(1, 5, 4, 0)
        >>> v2 -= v
        >>> v2
        Vector(0, 3, 4, 0)
        >>> v2 *= 10
        >>> v2
        Vector(0, 30, 40, 0)
        >>> v2 //= 10
        >>> v2
        Vector(0, 3, 4, 0)

        >>> B ** 0 == A
        True
        >>> B ** 1 == B
        True
        >>> B ** 2 == C
        True
        >>> B ** 3 == D
        True
        >>> B ** 4 == C - A
        True
        >>> B ** 5 == D - B
        True
        >>> B ** 6 == -A
        True
        >>> B ** 12 == A
        True

    """

    __slots__ = ('x0', 'x1', 'x2', 'x3')

    def __init__(self, *args: int):
        self.x0, self.x1, self.x2, self.x3 = args

    def copy(self) -> 'Vector':
        return Vector(*self.as_tuple())

    def as_tuple(self) -> VectorTuple:
        return self.x0, self.x1, self.x2, self.x3

    def __iter__(self):
        return iter(self.as_tuple())

    def __repr__(self):
        args = ', '.join(map(repr, self.as_tuple()))
        return f'{self.__class__.__name__}({args})'

    def __eq__(self, other: 'Vector') -> bool:
        return isinstance(other, Vector) and self.as_tuple() == other.as_tuple()

    def __neg__(self) -> 'Vector':
        return Vector(*map(operator.neg, self.as_tuple()))

    def __add__(self, other: Union['Vector', 'Transform']) -> Union['Vector', 'Transform']:
        if isinstance(other, Vector):
            return Vector(*map(operator.add, self.as_tuple(), other.as_tuple()))
        elif isinstance(other, Transform):
            return Transform(add=self) + other
        else:
            raise TypeError(type(other))

    def __sub__(self, other: Union['Vector', 'Transform']) -> Union['Vector', 'Transform']:
        if isinstance(other, Vector):
            return Vector(*map(operator.sub, self.as_tuple(), other.as_tuple()))
        elif isinstance(other, Transform):
            return Transform(add=self) - other
        else:
            raise TypeError(type(other))

    def __mul__(self, other: Union[int, 'Transform']) -> Union['Vector', 'Transform']:
        if isinstance(other, int):
            n = other
            return Vector(*map(operator.mul, self.as_tuple(), repeat(n)))
        elif isinstance(other, Transform):
            return Transform(add=self) * other
        else:
            raise TypeError(type(other))

    def __matmul__(self, other: 'Vector') -> 'Vector':
        return Vector(*_multiply(self.as_tuple(), other.as_tuple()))

    def __floordiv__(self, n: int) -> 'Vector':
        return Vector(*map(operator.floordiv, self.as_tuple(), repeat(n)))

    def __iadd__(self, other: Union['Vector', 'Transform']) -> Union['Vector', 'Transform']:
        if isinstance(other, Vector):
            self.x0 += other.x0
            self.x1 += other.x1
            self.x2 += other.x2
            self.x3 += other.x3
            return self
        elif isinstance(other, Transform):
            return Transform(add=self) + other
        else:
            raise TypeError(type(other))

    def __isub__(self, other: 'Vector') -> 'Vector':
        if isinstance(other, Vector):
            self.x0 -= other.x0
            self.x1 -= other.x1
            self.x2 -= other.x2
            self.x3 -= other.x3
            return self
        elif isinstance(other, Transform):
            return Transform(add=self) - other
        else:
            raise TypeError(type(other))

    def __imul__(self, other: Union[int, 'Transform']) -> Union['Vector', 'Transform']:
        if isinstance(other, int):
            n = other
            self.x0 *= n
            self.x1 *= n
            self.x2 *= n
            self.x3 *= n
            return self
        elif isinstance(other, Transform):
            return Transform(add=self) * other
        else:
            raise TypeError(type(other))

    def __imatmul__(self, other: 'Vector') -> 'Vector':
        self.x0, self.x1, self.x2, self.x3 = _multiply(
            self.as_tuple(), other.as_tuple())
        return self

    def __ifloordiv__(self, n: int) -> 'Vector':
        self.x0 //= n
        self.x1 //= n
        self.x2 //= n
        self.x3 //= n
        return self

    def __pow__(self, n: int) -> 'Vector':
        new_self = self.copy()
        new_self **= n
        return new_self

    def __ipow__(self, n: int) -> 'Vector':
        if n < 0:
            raise Exception(f"Negative power: {n!r}")
        if n == 0:
            self.x0 = 1
            self.x1 = self.x2 = self.x3 = 0
            return self
        else:
            orig_tup = tup = self.as_tuple()
            for i in range(n - 1):
                tup = _multiply(tup, orig_tup)
            self.x0, self.x1, self.x2, self.x3 = tup
            return self


class Transform:
    """

        >>> 

    """

    def __init__(self, flip: bool = False, rot: int = 0, add: Vector = None):
        self.flip = flip
        self.rot = rot
        self.add = add

    def __repr__(self):
        return f'{self.__class__.__name__}({self.flip!r}, {self.rot!r}, {self.add!r})'

    def __add__(self, v: Vector) -> 'Transform':
        new_self = self.copy()
        new_self += v
        return new_self

    def __sub__(self, v: Vector) -> 'Transform':
        new_self = self.copy()
        new_self.add -= v
        return new_self

    def __mul__(self, other: 'Transform') -> 'Transform':
        new_self = self.copy()
        new_self *= other
        return new_self

    def __imul__(self, other: 'Transform') -> 'Transform':
        if isinstance(other, Vector):
            if self.flip:
                other.flip()
            other.rot(self.rot)
            if self.add is not None:
                other += self.add
        elif isinstance(other, Transform):
            other.add = other.add.copy() if other.add else Zero.copy()
            if self.flip:
                other.flip = not other.flip
                other.rot = -other.rot
                other.add.flip()
            other.rot = other.rot + self.rot
            other.add.rot(self.rot)
            if other.add == Zero:
                other.add = None
        else:
            raise TypeError(type(other))


Zero = Vector(0, 0, 0, 0)
A = Vector(1, 0, 0, 0)
B = Vector(0, 1, 0, 0)
C = Vector(0, 0, 1, 0)
D = Vector(0, 0, 0, 1)
R = Transform(rot=1)
R2 = Transform(rot=2)
R3 = Transform(rot=3)
R4 = Transform(rot=4)
R5 = Transform(rot=5)
R6 = Transform(rot=6)
R7 = Transform(rot=7)
R8 = Transform(rot=8)
R9 = Transform(rot=9)
R10 = Transform(rot=10)
R11 = Transform(rot=11)
F = Transform(flip=True)


def render_vecs(vecs: Iterable[Vector], *, box: bool = True, char: str = 'x') -> str:
    """

        >>> print(render_vecs([
        ...     (Zero, '0'),
        ...     A, A * 2, (A * 3, 'A'),
        ...     B, B * 2, (B * 3, 'B'),
        ...     C, C * 2, (C * 3, 'C'),
        ...     D, D * 2, (D * 3, 'D'),
        ... ]))
        +-------+
        |D  C   |
        |       |
        |x x    |
        |      B|
        |xx  x  |
        |  x    |
        |0 x x A|
        +-------+

    """
    vecs = tuple((vec, char) if not isinstance(vec, tuple) else vec
        for vec in vecs)
    def to_coord(v: Vector) -> Tuple[int, int]:
        return 2 * (v.x0 + v.x1) + v.x2, -(v.x1 + 2 * (v.x2 + v.x3))
    coords = tuple(map(to_coord, (vec for vec, char in vecs)))
    if coords:
        min_x = min(x for x, y in coords)
        min_y = min(y for x, y in coords)
        max_x = max(x for x, y in coords)
        max_y = max(y for x, y in coords)
    else:
        min_x = min_y = max_x = max_y = 0
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
