import re
import string
from typing import List, Tuple, Union, Optional
from functools import wraps


MAX_DEPTH = 25
TOKEN_REGEX = re.compile(r'-?[1-9][0-9]*|[a-zA-Z]+|\$[a-zA-Z0-9_]+| +|.')


Term = Union[
    str,
    int,
    List['Term'],
    Tuple['Term', 'Term'],
]


class ParseError(Exception):
    pass


class Parser:
    """Parses syntax trees (called "terms") for a little language.

        >>> parser = Parser()
        >>> parser.parse('f')
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
        >>> parser.parse('(lf)')
        ['l', 'f']
        >>> parser.parse('()')
        []
        >>> parser.parse('()()')
        [([], [])]

        >>> parser.parse('$foo $bar')
        ['foo', 'bar']
        >>> parser.parse('$foo$bar')
        [('foo', 'bar')]
        >>> parser.parse('f$bar')
        [('f', 'bar')]

        >>> parser.parse('F=fr')
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
            if token[-1].isdigit():
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

        if '=' in line:
            # This line is a variable assignment
            name = next_token()
            if not isinstance(name, str):
                raise error(f"Can't use {name!r} as a variable name")
            next_token()
            expect('=')
            terms = get_terms(check_end=True)
            self.vars[name] = terms[0] if len(terms) == 1 else terms
            return []
        else:
            # Regular line consists of a list of terms
            next_token()
            return get_terms(check_end=True)


def main():
    """Little CLI program for testing"""

    def pprint(term, depth=0, indented=True):
        indent = '  ' * depth if indented else ''
        if isinstance(term, list):
            print(indent + ':')
            for child in term:
                pprint(child, depth + 1)
        elif isinstance(term, tuple):
            print(indent + str(term[0]) + '.', end='')
            pprint(term[1], depth, indented=False)
        else:
            print(indent + str(term))

    parser = Parser()
    while True:
        try:
            text = input('> ')
            if text and text[0] == '/':
                cmd = text[1:]
                if cmd == 'vars':
                    for name, term in parser.vars.items():
                        print(f'{name}=', end='')
                        pprint(term)
                elif cmd == 'exit':
                    return
                else:
                    print("Unrecognized command")
            else:
                terms = parser.parse(text)
                for term in terms:
                    pprint(term)
        except ParseError as ex:
            print(ex)
        except KeyboardInterrupt:
            print('bye!')
            return


if __name__ == '__main__':
    main()
