import re
import string
from inspect import isgeneratorfunction
from typing import List, Tuple, Union, Optional
from functools import wraps, partial, partialmethod
from contextlib import ExitStack, contextmanager

from geomant.geom import Vector, Zero, A, B, C, D, render_vecs


MAX_DEPTH = 25
TOKEN_REGEX = re.compile(r'-?[1-9][0-9]*|[a-zA-Z]+|\$[a-zA-Z0-9_]+| +|.')


Term = Union[
    str,
    int,
    List['Term'],
    Tuple['Term', 'Term'],
]


def pprint_term(term: Term, depth=0, indented=True):
    """Pretty-prints a term

        >>> pprint_term((3, 'f'))
        3.f
        >>> pprint_term((3, ['f', 'g']))
        3.:
          f
          g

    """
    indent = '  ' * depth if indented else ''
    if isinstance(term, list):
        print(indent + ':')
        for child in term:
            pprint_term(child, depth + 1)
    elif isinstance(term, tuple):
        print(indent + str(term[0]) + '.', end='')
        pprint_term(term[1], depth, indented=False)
    else:
        print(indent + str(term))


def flatten_term(term: Term) -> Term:
    """

        >>> flatten_term(3)
        3
        >>> flatten_term('f')
        'f'
        >>> flatten_term([1, 2])
        [1, 2]
        >>> flatten_term((1, 2))
        [1, 2]
        >>> flatten_term([(1, 2), [3, 4]])
        [1, 2, 3, 4]
        >>> flatten_term(((1, 2), [3, 4]))
        [1, 2, 3, 4]

    """
    if isinstance(term, (list, tuple)):
        terms = []
        for child in term:
            child = flatten_term(child)
            if isinstance(child, list):
                terms.extend(child)
            else:
                terms.append(child)
        return terms
    else:
        return term


class ParseError(Exception):
    pass


class Parser:
    """Parses syntax trees (called "terms") for a little language.

        >>> parser = Parser()
        >>> parser.parse('')
        []
        >>> parser.parse(' ')
        []
        >>> parser.parse('f')
        ['f']
        >>> parser.parse(' f ')
        ['f']
        >>> parser.parse('fl')
        ['f', 'l']
        >>> parser.parse('flf')
        ['f', 'l', 'f']
        >>> parser.parse('f lf')
        ['f', 'l', 'f']
        >>> parser.parse('-flf')
        [(-1, ['f', 'l', 'f'])]
        >>> parser.parse('f 3l')
        ['f', (3, 'l')]
        >>> parser.parse('f 3lf')
        ['f', (3, ['l', 'f'])]
        >>> parser.parse('f 3l f')
        ['f', (3, 'l'), 'f']
        >>> parser.parse('f3lf')
        [('f', (3, ['l', 'f']))]
        >>> parser.parse('f.lf')
        [('f', ['l', 'f'])]
        >>> parser.parse('f.l.f')
        [('f', ('l', 'f'))]
        >>> parser.parse('f(lf)')
        [('f', ['l', 'f'])]
        >>> parser.parse('f(l f)')
        [('f', ['l', 'f'])]
        >>> parser.parse('(lf)')
        ['l', 'f']
        >>> parser.parse('()')
        []
        >>> parser.parse('()()')
        [([], [])]
        >>> parser.parse(' ( )( ) ')
        [([], [])]
        >>> parser.parse(' ( ) ( ) ')
        []

        >>> parser.parse('$foo $bar')
        ['foo', 'bar']
        >>> parser.parse('$foo$bar')
        [('foo', 'bar')]
        >>> parser.parse('f$bar')
        [('f', 'bar')]

        >>> parser.parse('F=fr')
        []
        >>> parser.parse('F=fr;')
        []
        >>> parser.parse('FFl')
        ['f', 'r', 'f', 'r', 'l']
        >>> parser.parse('F.Fl')
        [(['f', 'r'], ['f', 'r', 'l'])]

        >>> parser.parse('G=3')
        []
        >>> parser.parse('G.fr')
        [(3, ['f', 'r'])]
        >>> parser.parse('Gfr')
        [3, 'f', 'r']

        >>> parser.parse('F=fr; G=3; F G')
        ['f', 'r', 3]
        >>> parser.parse('Z=;xZx')
        ['x', 'x']

        >>> parser.parse('Froz=$foo $bar')
        []
        >>> parser.parse('$x $Froz $y')
        ['x', 'foo', 'bar', 'y']
        >>> parser.parse('$x$Froz$y')
        [('x', (['foo', 'bar'], 'y'))]

    """

    def __init__(self):
        self.vars: Dict[str, Term] = {}

    def parse(self, text: str, **kwargs) -> List[Term]:
        return [term
            for row, line in enumerate(text.splitlines(), 1)
            for term in self.parse_line(row, line, **kwargs)]

    def parse_line(
            self,
            row: int,
            line: str,
            *,
            verbose: bool = False,
            max_depth: int = MAX_DEPTH,
            ) -> List[Term]:
        if verbose:
            print(f'parse_line: {line!r}')
        line = line.split('#', 1)[0] # strip comments
        matches = TOKEN_REGEX.finditer(line)
        match = None
        cur_token = None
        depth = 0

        def add_depth(func):
            @wraps(func)
            def wrapped_func(*args, **kwargs):
                nonlocal depth
                depth += 1
                if depth >= max_depth:
                    raise Exception(f"Parsed too deep: {depth}")
                try:
                    return func(*args, **kwargs)
                except Exception as ex:
                    if not isinstance(ex, ParseError):
                        ex = error(f"Error: {ex}")
                    raise ex
                finally:
                    depth -= 1
            return wrapped_func

        def next_token() -> Optional[str]:
            nonlocal match, cur_token
            match = next(matches, None)
            cur_token = match and match.group()
            if verbose:
                print('  ' * depth + f'next_token: {cur_token!r}')
            return cur_token

        def expect(expected: str):
            if cur_token != expected:
                raise error(f"Expected {expected!r}, got {cur_token!r}")
            next_token()

        def error(msg: str) -> ParseError:
            col = match.start() + 1 if match else 'end'
            return ParseError(f"[row={row}, col={col}, token={cur_token!r}]: {msg}")

        def get_var(name: str) -> Term:
            if not isinstance(name, str):
                raise error(f"Expected variable name, got: {name!r}")
            if name not in self.vars:
                raise error(f"Unset variable: {name!r}")
            return self.vars[name]

        @add_depth
        def get_term() -> Optional[Term]:
            if verbose:
                print('  ' * depth + 'get_term')

            # Consume leading whitespace
            while cur_token and cur_token[0] == ' ':
                next_token()

            # Make sure we have a token
            token = cur_token
            if cur_token is None:
                return None

            # Parse a term from the token!..
            if token[0].isdigit() or token[0] == '-' and token[-1].isdigit():
                term = int(token)
                next_token()
            elif token == '-':
                term = -1
                next_token()
            elif len(token) == 1 and token in string.ascii_uppercase:
                term = get_var(token)
                next_token()
            elif token[0] == '$':
                token = token[1:]
                if token[0] in string.ascii_uppercase:
                    term = get_var(token)
                else:
                    term = token
                next_token()
            elif token[0] in string.ascii_letters:
                terms = []
                for c in token:
                    if c in string.ascii_uppercase:
                        term = get_var(c)
                        if isinstance(term, list):
                            terms.extend(term)
                        else:
                            terms.append(term)
                    else:
                        terms.append(c)
                term = terms[0] if len(terms) == 1 else terms
                next_token()
            elif token == '(':
                next_token()
                term = get_terms()
                if len(term) == 1:
                    term = term[0]
                expect(')')
            else:
                return None

            if verbose:
                print('  ' * depth + f"...got term: {term!r}")

            # Parse term applications, if any
            while cur_token and cur_token[0] != ' ':
                rhs_term = get_term()
                if rhs_term is None and cur_token == '.':
                    next_token()
                    rhs_term = get_term()
                    if rhs_term is None:
                        raise error("Expected term after '.'")
                if rhs_term is None:
                    break
                term = (term, rhs_term)
                if verbose:
                    print('  ' * depth + f"...got term application: {term!r}")

            return term

        @add_depth
        def get_terms(check_end: bool = False) -> List[Term]:
            if verbose:
                print('  ' * depth + 'get_terms')
            terms = []
            while True:
                term = get_term()
                if term is None:
                    break
                if verbose:
                    print('  ' * depth + f"...got term: {term!r}")
                if isinstance(term, list):
                    terms.extend(term)
                else:
                    terms.append(term)
            if check_end:
                if cur_token is not None:
                    raise error(f"Expected end of line, found: {cur_token!r}")
            return terms

        # Prime the pump...
        next_token()

        # Handle variable assignments, if any
        while match and '=' in line[match.start():]:
            while cur_token[0] == ' ':
                next_token()
            name = cur_token
            if not isinstance(name, str) or name[0] not in string.ascii_uppercase:
                raise error(f"Can't use {name!r} as a variable name")
            next_token()
            expect('=')
            terms = get_terms()
            if cur_token is not None:
                expect(';')
            self.vars[name] = terms[0] if len(terms) == 1 else terms

        # Remainder of line consists of a list of terms
        return get_terms(check_end=True)

    def cli(self, **kwargs):
        yield from _cli(self, **kwargs)


def _cli(parser: Parser, *, input=input):
    r"""Little CLI program for testing

        >>> def input(prompt):
        ...     return inputs.pop(0)

        >>> inputs = [
        ...     'F=fg; G=3',
        ...     '\\v',
        ...     '\\v G',
        ...     '1.FG',
        ...     '\\q',
        ... ]
        >>> for term in Parser().cli(input=input):
        ...     pprint_term(term)
        F=:
          f
          g
        G=3
        G=3
        1.:
          f
          g
          3

    """
    while True:
        try:
            text = input('> ')
            if text and text[0] == '\\':
                cmd = text[1:]
                if cmd == 'v':
                    for name, term in parser.vars.items():
                        print(f'{name}=', end='')
                        pprint_term(term)
                elif cmd.startswith('v '):
                    name = cmd[2:]
                    term = parser.vars.get(name)
                    if term is None:
                        print(f"Unset variable: {name!r}")
                    else:
                        print(f'{name}=', end='')
                        pprint_term(term)
                elif cmd == 'q':
                    return
                else:
                    print(f"Unrecognized command: {cmd!r}")
            else:
                yield from parser.parse(text)
        except ParseError as ex:
            print(ex)
        except (EOFError, KeyboardInterrupt):
            print('bye!')
            return


class VectorTurtle:
    """

        >>> from string import ascii_lowercase

        >>> def parse(text):
        ...     vecs = []
        ...     turtle = VectorTurtle()
        ...     def do_p(turtle, term, char=None):
        ...         char = char or ascii_lowercase[len(vecs)]
        ...         vecs.append((turtle.pos.copy(), char))
        ...     turtle.atom('p')(do_p)
        ...     turtle.func('p')(do_p)
        ...     turtle.parse(text)
        ...     print(render_vecs(vecs))

        >>> parse('pfpfplfplfplfp')
        +--------+
        |       f|
        |        |
        |       e|
        |        |
        |      d |
        |a b c   |
        +--------+

        >>> parse('p f p(z) f p')
        +-----+
        |a z c|
        +-----+

        >>> parse('pfplfprbp')
        +-----+
        |  d c|
        |a b  |
        +-----+

        >>> parse('2pfplfp')
        +---------+
        |        c|
        |         |
        |a   b    |
        +---------+

        >>> parse('F=ff; pFplFp')
        +---------+
        |        c|
        |         |
        |a   b    |
        +---------+

        >>> parse('F=ff; F=FF; pFplFp')
        +-----------------+
        |                c|
        |                 |
        |                 |
        |                 |
        |a       b        |
        +-----------------+

        >>> parse('frf.pfplfp')
        +---------+
        |a       c|
        |    b    |
        +---------+

        >>> parse('2.frf.pfplfp')
        +-----------------+
        |a               c|
        |                 |
        |        b        |
        +-----------------+

        >>> parse('frf.ff.pfplfp')
        +-----------------+
        |a               c|
        |                 |
        |        b        |
        +-----------------+

        >>> parse('frf.2.pfplfp')
        +-----------------+
        |a               c|
        |                 |
        |        b        |
        +-----------------+

    """

    def __init__(self, parser: Parser = None):
        self.parser = parser or Parser()

        self.f = A.copy()
        self.pos = Zero.copy()
        self.atoms = {}
        self.funcs = {}

    def atom(self, name):
        def decorator(func):
            self.atoms[name] = func
            return func
        return decorator

    def func(self, name):
        def decorator(func):
            self.funcs[name] = func
            return func
        return decorator

    def parse(self, text: str):
        term = self.parser.parse(text)
        self.handle(term)

    def handle(self, term: Term):
        if isinstance(term, str):
            if term == 'f':
                self.pos += self.f
            elif term == 'b':
                self.pos -= self.f
            elif term == 'l':
                self.f @= B
            elif term == 'r':
                self.f @= B - D
            elif term in self.atoms:
                atom = self.atoms[term]
                atom(self, term)
            else:
                raise Exception(f"Don't know what to do with: {term!r}")
        elif isinstance(term, list):
            for child in term:
                self.handle(child)
        elif isinstance(term, tuple):
            func, arg = term
            if isinstance(func, int):
                n = func
                self.f *= n
                try:
                    self.handle(arg)
                finally:
                    self.f //= n
            elif isinstance(func, str) and func in self.funcs:
                name = func
                func = self.funcs[name]
                func(self, name, arg)
            else:
                old_pos, old_f = self.pos, self.f
                self.pos = Zero.copy()
                self.f = A.copy()
                try:
                    self.handle(func)
                    temp_f = self.pos
                finally:
                    self.pos, self.f = old_pos, old_f
                self.f = self.f @ temp_f
                try:
                    self.handle(arg)
                finally:
                    self.f = old_f
        else:
            raise Exception(f"Don't know what to do with: {term!r}")


if __name__ == '__main__':
    for term in Parser().cli():
        pprint_term(term)
