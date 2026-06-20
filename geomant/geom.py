import operator
from typing import Tuple
from itertools import repeat


class Vector:
    """

        >>> v = Vector(1, 2, 0, 0)
        >>> -v
        Vector(-1, -2, 0, 0)
        >>> 3 * v
        Vector(3, 6, 0, 0)

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

    """

    __slots__ = ('x0', 'x1', 'x2', 'x3')

    def __init__(self, *args: int):
        self.x0, self.x1, self.x2, self.x3 = args

    def copy(self) -> 'Vector':
        return Vector(*self.as_tuple())

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x0, self.x1, self.x2, self.x3

    def __iter__(self):
        return iter(self.as_tuple())

    def __repr__(self):
        args = ', '.join(map(str, self.as_tuple()))
        return f'{self.__class__.__name__}({args})'

    def __neg__(self) -> 'Vector':
        return Vector(*map(operator.neg, self.as_tuple()))

    def __rmul__(self, n: int) -> 'Vector':
        return Vector(*map(operator.mul, self.as_tuple(), repeat(n)))

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(*map(operator.add, self.as_tuple(), other.as_tuple()))

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(*map(operator.sub, self.as_tuple(), other.as_tuple()))

    def __iadd__(self, other: 'Vector') -> 'Vector':
        self.x0 += other.x0
        self.x1 += other.x1
        self.x2 += other.x2
        self.x3 += other.x3
        return self

    def __isub__(self, other: 'Vector') -> 'Vector':
        self.x0 -= other.x0
        self.x1 -= other.x1
        self.x2 -= other.x2
        self.x3 -= other.x3
        return self

    def __imul__(self, n: int) -> 'Vector':
        self.x0 *= n
        self.x1 *= n
        self.x2 *= n
        self.x3 *= n
        return self
