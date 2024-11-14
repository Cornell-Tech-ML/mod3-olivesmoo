"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Tracks history if required.

        Args:
        ----
            x (bool): If True, enables gradient tracking; if False, disables it.

        Returns:
        -------
            None

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if the tensor requires gradient tracking.

        Returns
        -------
            bool: True if the tensor requires gradients; False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a tensor filled with zeros.

        Args:
        ----
            shape (Optional[UserShape], optional): The desired shape of the
            output tensor. If None, uses the shape of the current tensor.

        Returns:
        -------
            Tensor: A tensor filled with zeros of the specified shape.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the tensor is a constant. A tensor with no history is a constant.

        Returns
        -------
            bool: True if the tensor is a constant (no history); False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of the tensor.

        Raises
        ------
            AssertionError: If the tensor has no history (i.e., it is constant).

        Returns
        -------
            Iterable[Variable]: An iterable containing the parent variables
            of the tensor.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the gradients for the chain rule of auto differentiation.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to the loss,
            used to compute the gradients of the input variables.

        Raises:
        ------
            AssertionError: If the tensor has no history, if the last function
            is not defined, if the context is missing, or if the number of
            inputs does not match the number of gradients returned.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing parent
            variables and its corresponding gradient.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Perform backpropagation to compute gradients.

        Args:
        ----
            grad_output (Optional[Tensor], optional): The gradient of the
            output with respect to the loss. If None and the tensor is not
            scalar, an assertion error will be raised.

        Raises:
        ------
            AssertionError: If `grad_output` is None and the tensor shape
            is not (1,), indicating that a non-scalar tensor requires a
            gradient output.

        Returns:
        -------
            None

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    # Functions
    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return int(operators.prod(self.shape))

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return len(self.shape)

    def __add__(self, b: TensorLike):
        b = self._ensure_tensor(b)
        return Add.apply(self, b)

    def __sub__(self, b: TensorLike):
        b_tensor = self._ensure_tensor(b)
        return Add.apply(self, Neg.apply(b_tensor))

    def __neg__(self):
        return Neg.apply(self)

    def __mul__(self, b: TensorLike):
        b = self._ensure_tensor(b)
        return Mul.apply(self, b)

    def __lt__(self, b: TensorLike):
        b = self._ensure_tensor(b)
        return LT.apply(self, b)

    def __eq__(self, b: TensorLike):
        b = self._ensure_tensor(b)
        return EQ.apply(self, b)

    def __gt__(self, b: TensorLike):
        b = self._ensure_tensor(b)
        return LT.apply(b, self)

    def __radd__(self, b: TensorLike):
        b = self._ensure_tensor(b)
        return Add.apply(b, self)

    def __rmul__(self, b: TensorLike):
        b = self._ensure_tensor(b)
        return Mul.apply(b, self)

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Compute the logical AND across dimensions of the tensor.

        Args:
        ----
            dim (Optional[int], optional): The dimension along which to compute
            the logical AND. If None, the operation is applied to the entire
            tensor.

        Returns:
        -------
            Tensor: A tensor containing the result of the logical AND operation.

        """
        if dim is None:
            return All.apply(self)
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, b: TensorLike) -> Tensor:
        """Check if two tensors are element-wise close to each other, within a tolerance.

        Args:
        ----
            b (TensorLike): The tensor or array-like object to compare against.

        Returns:
        -------
            Tensor: A tensor containing boolean values indicating whether
            each element of the current tensor is close to the corresponding
            element in `b`.

        """
        b = self._ensure_tensor(b)
        return IsClose.apply(self, b)

    def sigmoid(self) -> Tensor:
        """Compute the sigmoid activation function.

        Returns
        -------
            Tensor: A tensor containing the results of the sigmoid function
            applied to each element of the current tensor.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Compute the ReLU activation function.

        Returns
        -------
            Tensor: A tensor containing the results of the ReLU function
            applied to each element of the current tensor.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Compute the natural logarithm of the tensor.

        Returns
        -------
            Tensor: A tensor containing the natural logarithm of each element
            of the current tensor.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Compute the exponential of the tensor.

        Returns
        -------
            Tensor: A tensor containing the exponential of each element
            of the current tensor.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Compute the sum of the tensor elements along specified dimension.

        Args:
        ----
            dim (Optional[int], optional): The dimension along which to compute
            the sum. If None, the sum is computed over the entire tensor.

        Returns:
        -------
            Tensor: A tensor containing the sum of the elements along the specified
            dimension or the total sum if no dimension is provided.

        """
        if dim is None:
            return Sum.apply(self)
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean of the tensor elements along the specified dimension.

        Args:
        ----
            dim (Optional[int], optional): The dimension along which to compute
            the mean. If None, the mean is computed over the entire tensor.

        Returns:
        -------
            Tensor: A tensor containing the mean of the elements along the
            specified dimension or the overall mean if no dimension is provided.

        """
        if dim is None:
            return Sum.apply(self) * (1 / self.size)
        else:
            return Sum.apply(self, self._ensure_tensor(dim)) * (1 / self.shape[dim])

    def permute(self, *dim: int) -> Tensor:
        """Permute the dimensions of the tensor.

        This method rearranges the dimensions of the tensor according to
        the specified order. If no dimensions are provided, the original tensor
        is returned.

        Args:
        ----
            *dim (int): The new order of dimensions.

        Returns:
        -------
            Tensor: A tensor with its dimensions rearranged according to the
            specified order.

        """
        if len(dim) == 0:
            return self
        return Permute.apply(
            self, Tensor.make(list(dim), (len(dim),), backend=self.backend)
        )

    def view(self, *dim: int) -> Tensor:
        """Reshape the tensor without changing its data.

        This method returns a new tensor with the same data but with a
        different shape, specified by the dimensions provided. If no
        dimensions are given, the original tensor is returned.

        Args:
        ----
            *dim (int): The desired shape of the tensor.

        Returns:
        -------
            Tensor: A tensor with the specified shape, sharing the same data
            as the original tensor.

        """
        if len(dim) == 0:
            return self
        return View.apply(
            self, Tensor.make(list(dim), (len(dim),), backend=self.backend)
        )

    def zero_grad_(self) -> None:
        """Reset the gradient of the tensor to None.

        Returns
        -------
            None

        """
        self.grad = None
