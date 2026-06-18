from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, TypeVar, Iterator, Mapping

"""

    TODO: I think I realized that the get_affinity approach isn't going to
    work... but I forget why.
    I think it had something to do with the assumption I had when writing
    this file (made explicit in OneOfGrid.add_child) that the grids would
    be mutually exclusive?.. like, that there would be no overlap between
    the uhhhh the boundaries of the ArrayGrids?..
    Anyway, I think this whole file is probably going at things the wrong
    way.
    In particular, I think ArrayGrid is just not going to be useful,
    because with 4 dimensions, I think we end up with too much wasted
    space.

    I think we want to store tiles in a hierarchical structure very much
    like geom2018's "rendergraph", and then when doing lookups, we probably
    want a global dict mapping keys to tiles. Simple.

"""


GridKey = Tuple[int, int, int, int]
GridValue = TypeVar('GridValue')


class BaseGrid(Mapping[GridKey, GridValue], metaclass=ABCMeta):

    @abstractmethod
    def get_affinity(self, key: GridKey) -> float:
        """Returns a value between 0 and 1, representing how much sense it
        would make to store the given key in self.
        E.g. if storing the key would cost a lot of memory, then affinity
        should be lower."""

    @abstractmethod
    def get(self, key: GridKey) -> Optional[GridValue]: ...

    def __contains__(self, key: GridKey) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: GridKey) -> GridValue:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    @abstractmethod
    def __setitem__(self, key: GridKey, value: GridValue) -> None: ...

    @abstractmethod
    def delete(self, key: GridKey) -> bool: ...

    def __delitem__(self, key: GridKey) -> None:
        if not self.delete(key):
            raise KeyError(key)

    @abstractmethod
    def items(self) -> Iterator[Tuple[GridKey, GridValue]]: ...

    def __iter__(self) -> Iterator[GridKey]:
        for key, value in self.items():
            yield key

    keys = __iter__

    def values(self) -> Iterator[GridValue]:
        for key, value in self.items():
            yield value

    @abstractmethod
    def __len__(self) -> int: ...

    def update(self, other: 'BaseGrid') -> None:
        for key, value in other.items():
            self[key] = value


class SparseGrid(BaseGrid):
    """

        >>> grid = SparseGrid()
        >>> grid[0, 0, 0, 0] = 'A'
        >>> grid[0, 0, 0, 0]
        'A'
        >>> list(grid)
        [(0, 0, 0, 0)]
        >>> len(grid)
        1
        >>> del grid[0, 0, 0, 0]
        >>> list(grid)
        []

    """

    def __init__(self):
        self.tiles: Dict[GridKey, GridValue] = {}

    def get_affinity(self, key: GridKey) -> float:
        return .9

    def get(self, key: GridKey) -> Optional[GridValue]:
        return self.tiles.get(key)

    def __setitem__(self, key: GridKey, value: GridValue) -> None:
        self.tiles[key] = value

    def delete(self, key: GridKey) -> bool:
        if key not in self.tiles:
            return False
        del self.tiles[key]
        return True

    def items(self) -> Iterator[Tuple[GridKey, GridValue]]:
        return self.tiles.items()

    def __len__(self) -> int:
        return len(self.tiles)


class ArrayGrid(BaseGrid):
    """

        >>> grid = ArrayGrid(
        ...     min_key=(1, 1, 1, 1),
        ...     max_key=(3, 3, 3, 3),
        ... )

        >>> grid.dims
        (2, 2, 2, 2)

        >>> grid.size
        16

        >>> grid.within_bounds((1, 1, 1, 1))
        True
        >>> grid.within_bounds((0, 1, 1, 1))
        False
        >>> grid.within_bounds((3, 1, 1, 1))
        False
        >>> grid.within_bounds((0, 1, 1, 0))
        False
        >>> grid.within_bounds((2, 2, 2, 2))
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
        >>> len(grid)
        3

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

    def __init__(self, min_key: GridKey = None, max_key: GridKey = None):
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

    def get_affinity(self, key: GridKey) -> float:
        # TODO: if not within bounds, get a distance from our boundary and
        # use that to return an affinity between 0 and 1
        return 1 if self.within_bounds(key) else .5

    def within_bounds(self, key: GridKey) -> bool:
        k0, k1, k2, k3 = key
        a0, a1, a2, a3 = self.min_key
        if k0 < a0 or k1 < a1 or k2 < a2 or k3 < a3:
            return False
        b0, b1, b2, b3 = self.max_key
        if k0 >= b0 or k1 >= b1 or k2 >= b2 or k3 >= b3:
            return False
        return True

    def get(self, key: GridKey) -> Optional[GridValue]:
        if not self.within_bounds(key):
            return None
        i = self.get_index(key)
        return self.tiles[self.get_index(key)]

    def _expand_tiles(self, key: GridKey):
        k0, k1, k2, k3 = key
        a0, a1, a2, a3 = self.min_key
        b0, b1, b2, b3 = self.max_key
        new_min_key = (min(k0, a0), min(k1, a1), min(k2, a2), min(k3, a3))
        new_max_key = (max(k0 + 1, b0), max(k1 + 1, b1), max(k2 + 1, b2), max(k3 + 1, b3))

        new_grid = ArrayGrid(new_min_key, new_max_key)
        new_grid.update(self)

        self.min_key = new_min_key
        self.max_key = new_max_key
        self.dims = new_grid.dims
        self.size = new_grid.size
        self.tiles = new_grid.tiles

    def __setitem__(self, key: GridKey, value: GridValue) -> None:
        if not self.within_bounds(key):
            self._expand_tiles(key)
        i = self.get_index(key)
        self.tiles[i] = value

    def delete(self, key: GridKey) -> bool:
        if not self.within_bounds(key):
            return False
        i = self.get_index(key)
        if not self.tiles[i]:
            raise False
        self.tiles[i] = None
        return True

    def items(self) -> Iterator[Tuple[GridKey, GridValue]]:
        for i, tile in enumerate(self.tiles):
            if tile is not None:
                yield self.get_key(i), tile

    def __len__(self) -> int:
        return sum(1 for tile in self.tiles if tile is not None)


class OneOfGrid(BaseGrid):
    """

        >>> array_grid = ArrayGrid(max_key=(1, 1, 1, 5))
        >>> array_grid[0, 0, 0, 0] = 'A'
        >>> sparse_grid = SparseGrid()
        >>> sparse_grid[1, 0, 0, 0] = 'B'
        >>> grid = OneOfGrid(array_grid, sparse_grid)
        >>> list(grid.values())
        ['A', 'B']

        >>> array_grid2 = ArrayGrid()
        >>> array_grid2[0, 0, 0, 0] = 'A'
        >>> grid.add_child(array_grid2)
        Traceback (most recent call last):
         ...
        Exception: Can't add ... already has key (0, 0, 0, 0)

    """

    def __init__(self, *children):
        self.children = []
        for child in children:
            self.add_child(child)

    def add_child(self, new_child: BaseGrid) -> None:
        new_keys = set(new_child)
        for child in self.children:
            bad_key = next((key for key in child if key in new_keys), None)
            if bad_key:
                raise Exception(f"Can't add {new_child} to {self}: child {child} already has key {bad_key}")
        self.children.append(new_child)

    def get_affinity(self, key: GridKey) -> float:
        return max(child.get_affinity(key) for child in self.children)

    def get(self, key: GridKey) -> Optional[GridValue]:
        for child in self.children:
            value = child.get(key)
            if value is not None:
                return value
        return None

    def __setitem__(self, key: GridKey, value: GridValue) -> None:
        if not self.children:
            raise KeyError(key)
        child = max(self.children, lambda child: child.get_affinity(key))
        child[key] = value

    def delete(self, key: GridKey) -> bool:
        for child in self.children:
            if child.delete(key):
                return True
        return False

    def items(self) -> Iterator[Tuple[GridKey, GridValue]]:
        for child in self.children:
            yield from child.items()

    def __len__(self) -> int:
        return sum(map(len(self.children)))
