from typing import Tuple, TypeAlias, Union

from jaxtyping import Array, ArrayLike, Num

_Axis: TypeAlias = Union[None, int, Tuple[int, ...]]
_ArrayLike: TypeAlias = Num[ArrayLike, " ..."]
_Array: TypeAlias = Num[Array, " ..."]

def logsumexp(a: _ArrayLike, axis: _Axis = None) -> _Array: ...
