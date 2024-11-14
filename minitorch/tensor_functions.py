"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple, Optional, Union

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for negation.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for negation.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient of the input tensor.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for inverse.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The inverse of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for inverse.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for addition.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The sum of the input tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for addition.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the input tensors.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The product of the input tensors.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the input tensors.

        """
        (
            t1,
            t2,
        ) = ctx.saved_values
        grad_t1 = grad_output.f.mul_zip(t2, grad_output)
        grad_t2 = grad_output.f.mul_zip(t1, grad_output)
        return grad_t1, grad_t2


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        sigma = t1.f.sigmoid_map(t1)
        neg_sigma = grad_output.f.neg_map(sigma)
        mult_sigma = grad_output.f.mul_zip(
            sigma, grad_output.f.add_zip(t1._ensure_tensor(1), neg_sigma)
        )
        return grad_output.f.mul_zip(mult_sigma, grad_output)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The output tensor after applying ReLU.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the logarithm function.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The natural logarithm of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the logarithm function.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The exponential of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        ex = t1.f.exp_map(t1)
        return grad_output.f.mul_zip(ex, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Compute the forward pass for summation.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.
            dim (Optional[Tensor]): The dimension along which to sum.

        Returns:
        -------
            Tensor: The sum of the input tensor along the specified dimension.

        """
        ctx.save_for_backward(t1, dim)
        if dim is None:
            for _ in range(len(t1.shape)):
                t1 = t1.f.add_reduce(t1, 0)
                if len(t1.shape) > 1:
                    t1 = minitorch.Tensor.make(
                        t1._tensor._storage, t1.shape[1:], backend=t1.backend
                    )
            return t1
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(
        ctx: Context, grad_output: Tensor
    ) -> Union[Tensor, Tuple[Tensor, float]]:
        """Compute the backward pass for summation.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Union[Tensor, Tuple[Tensor, float]]: The gradient of the input tensor or a tuple
            with a gradient and a float indicating the dimension.

        """
        (t1, dim) = ctx.saved_values
        output = grad_output.f.add_zip(zeros(t1.shape, t1.backend), grad_output)
        if dim is not None:
            return (output, 0.0)
        else:
            return output


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for the less than operation.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A boolean tensor indicating if elements of t1 are less than elements of t2.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for the less than operation.

        Args:
        ----
            ctx (Context): The context containing saved information.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the input tensors.

        """
        (t1, t2) = ctx.saved_tensors
        grad_t1 = zeros(t1.shape)
        grad_t2 = zeros(t2.shape)

        return grad_t1, grad_t2


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for the equality operation.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A boolean tensor indicating if elements of t1 are equal to elements in t2.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for the equality operation.

        Args:
        ----
            ctx (Context): The context containing information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the equality operation.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to the input tensors t1 and t2, both initialized to zero.

        """
        (t1, t2) = ctx.saved_tensors
        grad_t1 = zeros(t1.shape)
        grad_t2 = zeros(t2.shape)

        return grad_t1, grad_t2


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute element-wise comparison for closeness between two tensors.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A boolean tensor indicating whether elements of t1 are close to elements of t2.

        """
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of the input tensor according to the specified order.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor to permute.
            order (Tensor): A tensor specifying the order of the dimensions.

        Returns:
        -------
            Tensor: A new tensor with its dimensions permuted according to the specified order.

        """
        ctx.save_for_backward(t1, order)
        order_items = order._tensor._storage.tolist()
        order_items = list(map(int, order_items))
        permuted_data = t1._tensor.permute(*order_items)
        return minitorch.Tensor(permuted_data, backend=t1.backend)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the backward pass for the permute operation.

        Args:
        ----
            ctx (Context): The context containing information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the permute operation.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient of the loss with respect to the input tensor t1,
            and a placeholder for the order gradient (set to zero).

        """
        (t1, order) = ctx.saved_values

        order_items = order._tensor._storage.tolist()
        order_items = list(map(int, order_items))

        reverse_order = [0] * len(order_items)
        for i, item in enumerate(order_items):
            reverse_order[item] = i

        unpermuted_grad = minitorch.Tensor(
            grad_output._tensor.permute(*reverse_order), backend=grad_output.backend
        )
        output = unpermuted_grad.f.add_zip(zeros(t1.shape, t1.backend), unpermuted_grad)

        return (output, 0.0)


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Create a new view of the input tensor with the specified shape.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            a (Tensor): The input tensor to be viewed.
            shape (Tensor): A tensor specifying the new shape for the view.

        Returns:
        -------
            Tensor: A new tensor that shares storage with the original tensor
            but has the specified shape.

        Raises:
        ------
            AssertionError: If the input tensor is not contiguous in memory.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the gradient of a function using central difference approximation.

    Args:
    ----
        f (Any): The function for which the gradient is being computed.
        *vals (Tensor): The input tensors for the function `f`.
        arg (int, optional): The index of the argument with respect to which
                             the gradient is computed. Defaults to 0.
        epsilon (float, optional): A small value used to compute the difference.
                                    Defaults to 1e-6.
        ind (UserIndex): The specific index in the tensor to perturb.

    Returns:
    -------
        float: The estimated gradient of the function with respect to the
               specified argument.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
