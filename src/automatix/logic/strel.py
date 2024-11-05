import itertools
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from lark import Lark, Token, Transformer, ast_utils, v_args

STREL_GRAMMAR_FILE = Path(__file__).parent / "strel.lark"


class _Ast(ast_utils.Ast):
    pass


class _Phi(_Ast):
    pass


@dataclass
class TimeInterval(_Ast):
    start: Optional[int]
    end: Optional[int]


@dataclass
class DistanceInterval(_Ast):
    start: Optional[float]
    end: Optional[float]


@dataclass
class Identifier(_Phi):
    name: str


@dataclass
class NotOp(_Phi):
    arg: _Phi


@dataclass
class AndOp(_Phi):
    lhs: _Phi
    rhs: _Phi


@dataclass
class OrOp(_Phi):
    lhs: _Phi
    rhs: _Phi


@dataclass
class EverywhereOp(_Phi):
    interval: DistanceInterval
    arg: _Phi


@dataclass
class SomewhereOp(_Phi):
    interval: DistanceInterval
    arg: _Phi


@dataclass
class EscapeOp(_Phi):
    interval: DistanceInterval
    arg: _Phi


@dataclass
class ReachOp(_Phi):
    lhs: _Phi
    interval: DistanceInterval
    rhs: _Phi


@dataclass
class NextOp(_Phi):
    interval: Optional[TimeInterval]
    arg: _Phi


@dataclass
class GloballyOp(_Phi):
    interval: Optional[TimeInterval]
    arg: _Phi


@dataclass
class EventuallyOp(_Phi):
    interval: Optional[TimeInterval]
    arg: _Phi


@dataclass
class UntilOp(_Phi):
    lhs: _Phi
    interval: Optional[TimeInterval]
    rhs: _Phi


class _TransformTerminals(Transformer):

    def CNAME(self, s: Token) -> str:  # noqa: N802
        return str(s)

    def ESCAPED_STRING(self, s: Token) -> str:  # noqa: N802
        # Remove quotation marks
        return s[1:-1]

    def INT(self, tok: Token) -> int:  # noqa: N802
        return int(tok)

    def NUMBER(self, tok: Token) -> float:  # noqa: N802
        return float(tok)

    @v_args(inline=True)
    def phi(self, x: Token) -> Token:
        return x


@lru_cache(maxsize=1)
def get_parser() -> Lark:
    with open(STREL_GRAMMAR_FILE, "r") as grammar:
        return Lark(
            grammar,
            start="phi",
            strict=True,
        )


@lru_cache(maxsize=1)
def _to_ast_transformer() -> Transformer:
    ast = types.ModuleType("ast")
    for c in itertools.chain(_Ast.__subclasses__(), _Phi.__subclasses__()):
        ast.__dict__[c.__name__] = c
    return ast_utils.create_transformer(ast, _TransformTerminals())


TO_AST_TRANSFORMER = _to_ast_transformer()


def parse(expr: str) -> _Ast:
    tree = get_parser().parse(expr)

    return TO_AST_TRANSFORMER.transform(tree)
