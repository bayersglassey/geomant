import operator
from typing import Tuple, Union
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

        When vectors are multiplied, the unit vector B acts as a 30 degree
        counter-clockwise rotation, i.e. B^0 = A, B^1 = B, B^2 = C, B^3 = D.

        >>> v = Vector(1, 2, 0, 0)
        >>> -v
        Vector(-1, -2, 0, 0)
        >>> 3 * v
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
        return self.as_tuple() == other.as_tuple()

    def __neg__(self) -> 'Vector':
        return Vector(*map(operator.neg, self.as_tuple()))

    def __mul__(self, other: Union[int, 'Vector']) -> 'Vector':
        if isinstance(other, int):
            n = other
            return Vector(*map(operator.mul, self.as_tuple(), repeat(n)))
        else:
            return Vector(*_multiply(self.as_tuple(), other.as_tuple()))

    __rmul__ = __mul__

    def __floordiv__(self, n: int) -> 'Vector':
        return Vector(*map(operator.floordiv, self.as_tuple(), repeat(n)))

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

    def __imul__(self, other) -> 'Vector':
        if isinstance(other, int):
            n = other
            self.x0 *= n
            self.x1 *= n
            self.x2 *= n
            self.x3 *= n
        else:
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
        if n < 0:
            raise Exception(f"Negative power: {n!r}")
        if n == 0:
            return Vector(1, 0, 0, 0)
        other = self.copy()
        for i in range(n - 1):
            other *= self
        return other


A = Vector(1, 0, 0, 0)
B = Vector(0, 1, 0, 0)
C = Vector(0, 0, 1, 0)
D = Vector(0, 0, 0, 1)
neg_A = -A
neg_B = -B
neg_C = -C
neg_D = -D
