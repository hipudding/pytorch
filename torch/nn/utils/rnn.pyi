from typing import (
    Any,
    Iterable,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    Union,
    Self as typeingSelf,
)

from typing_extensions import Self

from torch import Tensor

from torch._prims_common import DeviceLikeType
from torch.types import _dtype

class PackedSequence_(NamedTuple):
    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Optional[Tensor]
    unsorted_indices: Optional[Tensor]

def bind(optional: Any, fn: Any): ...

class PackedSequence(PackedSequence_):
    def __new__(
        cls,
        data: Tensor,
        batch_sizes: Optional[Tensor] = ...,
        sorted_indices: Optional[Tensor] = ...,
        unsorted_indices: Optional[Tensor] = ...,
    ) -> Self: ...
    def pin_memory(self) -> typeingSelf: ...
    def cuda(self, *args: Any, **kwargs: Any) -> typeingSelf: ...
    def cpu(self) -> typeingSelf: ...
    def double(self) -> typeingSelf: ...
    def float(self) -> typeingSelf: ...
    def half(self) -> typeingSelf: ...
    def long(self) -> typeingSelf: ...
    def int(self) -> typeingSelf: ...
    def short(self) -> typeingSelf: ...
    def char(self) -> typeingSelf: ...
    def byte(self) -> typeingSelf: ...
    @overload
    def to(
        self,
        dtype: _dtype,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> typeingSelf: ...
    @overload
    def to(
        self,
        device: Optional[DeviceLikeType] = None,
        dtype: Optional[_dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> typeingSelf: ...
    @overload
    def to(
        self,
        other: Tensor,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> typeingSelf: ...
    @property
    def is_cuda(self) -> bool: ...
    def is_pinned(self) -> bool: ...

def invert_permutation(permutation: Optional[Tensor]): ...
def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor,
    batch_first: bool = ...,
    enforce_sorted: bool = ...,
) -> PackedSequence: ...
def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = ...,
    padding_value: float = ...,
    total_length: Optional[int] = ...,
) -> Tuple[Tensor, ...]: ...
def pad_sequence(
    sequences: Union[Tensor, Iterable[Tensor]],
    batch_first: bool = False,
    padding_value: float = ...,
) -> Tensor: ...
def pack_sequence(
    sequences: Sequence[Tensor],
    enforce_sorted: bool = ...,
) -> PackedSequence: ...
def get_packed_sequence(
    data: Tensor,
    batch_sizes: Optional[Tensor],
    sorted_indices: Optional[Tensor],
    unsorted_indices: Optional[Tensor],
) -> PackedSequence: ...
