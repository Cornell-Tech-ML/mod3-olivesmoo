# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile the provided function for device execution using optional keyword arguments.

    Args:
    ----
        fn: The function to JIT compile. This can be any callable.
        **kwargs: Optional keyword arguments for the JIT compiler.

    Returns:
    -------
        Callable[..., Any]: A callable function representing the JIT compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable[..., Any], **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile the provided function using optional keyword arguments.

    Args:
    ----
        fn: The function to JIT compile. This can be any callable.
        **kwargs: Optional keyword arguments for the JIT compiler.

    Returns:
    -------
        FakeCUDAKernel: A fake CUDA kernel (or a callable that represents the JIT compiled function).

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Creates a function that applies a binary operation to corresponding elements of two tensors
        element-wise, using a CUDA kernel.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that takes two floats and returns
                                                a float. This function will be applied element-wise
                                                to the tensors `a` and `b`.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that, when called with two tensors `a` and `b`,
                                                returns a tensor where each element is the result of applying
                                                the binary operation `fn` to the corresponding elements of `a` and `b`.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Creates a reduction function that applies a binary operation to the elements of a tensor
        along a specified dimension using a CUDA kernel.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that takes two floats and
                                                returns a float. This function is used to reduce
                                                the elements of the tensor.
            start (float, optional): The initial value for the reduction. Default is 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that, when called with a tensor `a` and a
                                            dimension `dim`, returns a tensor containing the result
                                            of applying the reduction along that dimension.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication for tensors, supporting both 2D and higher-dimensional inputs.

        Args:
        ----
            a (Tensor): The first input tensor with shape `(batch_size, m, n)`, where `m` and `n`
                        are the dimensions of the matrix to multiply.
            b (Tensor): The second input tensor with shape `(batch_size, n, p)`, where `n` is the
                        matching dimension for matrix multiplication and `p` is the number of columns
                        in the resulting matrix.

        Returns:
        -------
            Tensor: A tensor containing the result of the matrix multiplication with shape
                    `(batch_size, m, p)`.

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """A practice kernel for summing elements within blocks, preparing for reduce operations.

    This kernel takes an input array `a` of length `size` and an output array `out`
    of size `size // blockDim`. It computes the sum of elements within each block
    and stores the result in the corresponding cell of the `out` array.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0

    cuda.syncthreads()

    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride *= 2
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Performs a sum operation on the input tensor `a` using a CUDA kernel.

    This function divides the input tensor `a` into blocks, computes the sum of elements
    within each block, and stores the results in the output tensor `out`. The computation
    is done on the GPU using the CUDA kernel `jit_sum_practice`.

    Args:
    ----
        a (Tensor): The input tensor with shape `(size,)`, which contains the elements
                    to be summed.

    Returns:
    -------
        TensorData: A tensor containing the result of the sum operation, with shape `(2,)`.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # assign a block to each index reduced over
        reduce_size = a_shape[reduce_dim]
        if pos < reduce_size:
            to_index(out_pos, out_shape, out_index)  # index of output
            j = index_to_position(
                out_index, a_strides
            )  # first position of the dim we want to accumulate
            offset = a_strides[reduce_dim] * pos
            cache[pos] = a_storage[j + offset]
        else:
            cache[pos] = reduce_value

        cuda.syncthreads()
        stride = 1
        while stride < BLOCK_DIM:
            if pos % (2 * stride) == 0:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride *= 2
        if pos == 0:
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y

    if i < size and j < size:
        a_shared[local_i, local_j] = a[i * size + j]  # a[i, j]
        b_shared[local_i, local_j] = b[i * size + j]  # b[i, j]

        cuda.syncthreads()

        acc = 0.0
        for k in range(size):
            acc += a_shared[local_i, k] * b_shared[k, local_j]
        out[i * size + j] = acc


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs matrix multiplication between two tensors `a` and `b` using CUDA kernels.

    This function initializes an output tensor `out` with the appropriate size and
    moves it to the GPU. It then invokes a CUDA kernel `jit_mm_practice` to perform
    matrix multiplication of the input tensors on the GPU.

    Args:
    ----
        a (Tensor): The first input tensor with shape `(size, size)`. Represents the
            left-hand matrix in the multiplication.
        b (Tensor): The second input tensor with shape `(size, size)`. Represents the
            right-hand matrix in the multiplication.

    Returns:
    -------
        TensorData: A tensor containing the result of the matrix multiplication
        with shape `(size, size)`.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.

    acc = 0
    size = a_shape[-1]  # should equal b_shape[-2]
    for k in range(0, size, BLOCK_DIM):
        if batch < out_shape[0] and i < a_shape[-2] and k + pj < size:
            a_index = (
                batch * a_batch_stride + i * a_strides[1] + (k + pj) * a_strides[2]
            )
            a_shared[pi, pj] = a_storage[a_index]
            # a_shared[pi, pj] = a[i, k + pj]
        if batch < out_shape[0] and j < b_shape[-1] and k + pi < size:
            b_index = (
                batch * b_batch_stride + (k + pi) * b_strides[1] + j * b_strides[2]
            )
            b_shared[pi, pj] = b_storage[b_index]
            # b_shared[pi, pj] = b[k + pi, j]

        cuda.syncthreads()

        for local_k in range(min(BLOCK_DIM, size - k)):
            acc += a_shared[pi, local_k] * b_shared[local_k, pj]

    if batch < out_shape[0] and i < out_shape[-2] and j < out_shape[-1]:
        out_index = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        out[out_index] = acc

    # def mm_shared1(out, a, b, K):
    # ...
    # for s in range(0, K, TPB):
    #     sharedA[local_i, local_j] = a[i, s + local_j]
    #     sharedB[local_i, local_j] = b[s + local_i, j]
    #     ...
    #     for k in range(TPB):
    #         t += sharedA[local_i, k] * sharedB[k, local_j]
    # out[i, j] = t
    # raise NotImplementedError("Need to implement for Task 3.4")


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
