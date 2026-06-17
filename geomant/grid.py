from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, TypeVar, Iterable, Mapping


GridKey = Tuple[int, int, int, int]
GridValue = TypeVar('GridValue')


class BaseTileGrid(Mapping[GridKey, GridValue], metaclass=ABCMeta):

    @abstractmethod
    def get(self, key: GridKey) -> Optional[GridValue]: ...

    def __getitem__(self, key: GridKey) -> GridValue:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    @abstractmethod
    def __setitem__(self, key: GridKey, value: GridValue) -> None: ...

    @abstractmethod
    def __delitem__(self, key: GridKey) -> None: ...

    @abstractmethod
    def items(self) -> Iterable[Tuple[GridKey, GridValue]]: ...

    def __iter__(self) -> Iterable[GridKey]:
        for key, value in self.items():
            yield key

    keys = __iter__

    def values(self) -> Iterable[GridValue]:
        for key, value in self.items():
            yield value

    @abstractmethod
    def __len__(self) -> int: ...

    def update(self, other: 'BaseTileGrid') -> None:
        for key, value in other.items():
            self[key] = value


class ArrayTileGrid(BaseTileGrid):
    """

        >>> grid = ArrayTileGrid(
        ...     min_key=(1, 1, 1, 1),
        ...     max_key=(3, 3, 3, 3),
        ... )

        >>> grid.dims
        (2, 2, 2, 2)

        >>> grid.size
        16

        >>> grid.contains((1, 1, 1, 1))
        True
        >>> grid.contains((0, 1, 1, 1))
        False
        >>> grid.contains((3, 1, 1, 1))
        False
        >>> grid.contains((0, 1, 1, 0))
        False
        >>> grid.contains((2, 2, 2, 2))
        True

        >>> grid.get_index((1, 1, 1, 1))
        0
        >>> grid.get_index((1, 1, 1, 2))
        1
        >>> grid.get_index((2, 1, 1, 1))
        8
        >>> grid.get_index((2, 2, 2, 2))
        15

        >>> grid.get_key(0)
        (1, 1, 1, 1)
        >>> grid.get_key(1)
        (1, 1, 1, 2)
        >>> grid.get_key(8)
        (2, 1, 1, 1)
        >>> grid.get_key(15)
        (2, 2, 2, 2)

        >>> grid[1, 1, 1, 1] = 'A'
        >>> grid[1, 2, 2, 1] = 'B'
        >>> grid[2, 2, 2, 2] = 'C'
        >>> grid.size
        16

        >>> list(grid)
        [(1, 1, 1, 1), (1, 2, 2, 1), (2, 2, 2, 2)]
        >>> list(grid.values())
        ['A', 'B', 'C']

        >>> grid.get((1, 2, 2, 1))
        'B'
        >>> grid.get((2, 1, 1, 1))
        >>> grid.get((3, 3, 3, 3))

        >>> grid[0, 1, 1, 1] = 'D'
        >>> grid.min_key
        (0, 1, 1, 1)
        >>> grid.size
        24
        >>> for k, v in grid.items(): print(f"{k}: {v}")
        (0, 1, 1, 1): D
        (1, 1, 1, 1): A
        (1, 2, 2, 1): B
        (2, 2, 2, 2): C

        >>> grid[3, 1, 1, 1] = 'E'
        >>> grid.max_key
        (4, 3, 3, 3)
        >>> grid.size
        32
        >>> for k, v in grid.items(): print(f"{k}: {v}")
        (0, 1, 1, 1): D
        (1, 1, 1, 1): A
        (1, 2, 2, 1): B
        (2, 2, 2, 2): C
        (3, 1, 1, 1): E

    """

    def __init__(self, min_key: GridKey, max_key: GridKey):
        self.min_key = min_key or (0, 0, 0, 0)
        self.max_key = max_key or (0, 0, 0, 0)
        self._update_dims_and_size()
        self.tiles = [None] * self.size

    def _update_dims_and_size(self):
        a0, a1, a2, a3 = self.min_key
        b0, b1, b2, b3 = self.max_key
        d0 = b0 - a0; d1 = b1 - a1; d2 = b2 - a2; d3 = b3 - a3
        self.dims = (d0, d1, d2, d3)
        self.size = d0 * d1 * d2 * d3

    def get_index(self, key: GridKey) -> int:
        k0, k1, k2, k3 = key
        a0, a1, a2, a3 = self.min_key
        k0 -= a0; k1 -= a1; k2 -= a2; k3 -= a3
        d0, d1, d2, d3 = self.dims
        return k3 + d3 * (k2 + d2 * (k1 + d1 * k0))

    def get_key(self, i: int) -> GridKey:
        d0, d1, d2, d3 = self.dims
        k3 = i % d3
        i //= d3
        k2 = i % d2
        i //= d2
        k1 = i % d1
        i //= d1
        k0 = i

        a0, a1, a2, a3 = self.min_key
        return (k0 + a0, k1 + a1, k2 + a2, k3 + a3)

    def contains(self, key: GridKey) -> bool:
        k0, k1, k2, k3 = key
        a0, a1, a2, a3 = self.min_key
        if k0 < a0 or k1 < a1 or k2 < a2 or k3 < a3:
            return False
        b0, b1, b2, b3 = self.max_key
        if k0 >= b0 or k1 >= b1 or k2 >= b2 or k3 >= b3:
            return False
        return True

    def get(self, key: GridKey) -> Optional[GridValue]:
        if self.contains(key):
            i = self.get_index(key)
            return self.tiles[self.get_index(key)]
        else:
            return None

    def _expand_tiles(self, key: GridKey):
        k0, k1, k2, k3 = key
        a0, a1, a2, a3 = self.min_key
        b0, b1, b2, b3 = self.max_key
        new_min_key = (min(k0, a0), min(k1, a1), min(k2, a2), min(k3, a3))
        new_max_key = (max(k0 + 1, b0), max(k1 + 1, b1), max(k2 + 1, b2), max(k3 + 1, b3))

        new_grid = ArrayTileGrid(new_min_key, new_max_key)
        new_grid.update(self)

        self.min_key = new_min_key
        self.max_key = new_max_key
        self.dims = new_grid.dims
        self.size = new_grid.size
        self.tiles = new_grid.tiles

    def __setitem__(self, key: GridKey, value: GridValue) -> None:
        if not self.contains(key):
            self._expand_tiles(key)
        i = self.get_index(key)
        self.tiles[i] = value

    def __delitem__(self, key: GridKey) -> None:
        if self.contains(key):
            i = self.get_index(key)
            self.tiles[i] = None
        else:
            raise KeyError(key)

    def items(self) -> Iterable[Tuple[GridKey, GridValue]]:
        for i, tile in enumerate(self.tiles):
            if tile is not None:
                yield self.get_key(i), tile

    def __len__(self) -> int:
        return sum(1 for tile in self.tiles if tile is not None)
