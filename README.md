# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Module 3.1 and 3.2
```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/oliviamei/cornell_stuff/mle_codebase/mod3-olivesmoo/minitorch/fast_ops.py
 (165)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/oliviamei/cornell_stuff/mle_codebase/mod3-olivesmoo/minitorch/fast_ops.py (165)
--------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                     |
        out: Storage,                                                                             |
        out_shape: Shape,                                                                         |
        out_strides: Strides,                                                                     |
        in_storage: Storage,                                                                      |
        in_shape: Shape,                                                                          |
        in_strides: Strides,                                                                      |
    ) -> None:                                                                                    |
        # TODO: Implement for Task 3.1.                                                           |
        if np.array_equal(in_strides, out_strides) and np.array_equal(in_shape, out_shape):       |
            for i in prange(len(out)):------------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                        |
        else:                                                                                     |
            in_indices = np.empty((len(out), len(in_shape)), dtype=np.int32)                      |
            out_indices = np.empty((len(out), len(out_shape)), dtype=np.int32)                    |
            for i in prange(len(out)):------------------------------------------------------------| #1
                to_index(i, out_shape, out_indices[i])                                            |
                broadcast_index(out_indices[i], out_shape, in_shape, in_indices[i])               |
                o = index_to_position(out_indices[i], out_strides)                                |
                j = index_to_position(in_indices[i], in_strides)                                  |
                out[o] = fn(in_storage[j])                                                        |
            #     print(out[o])                                                                   |
            # print(out)                                                                          |
                                                                                                  |
        '''                                                                                       |
            out_index: Index = np.zeros(MAX_DIMS, np.int32)                                       |
            in_index: Index = np.zeros(MAX_DIMS, np.int32)                                        |
            for i in range(len(out)):                                                             |
                to_index(i, out_shape, out_index) #get the out index given ordinal + out shape    |
                broadcast_index(out_index, out_shape, in_shape, in_index) # get the in index      |
                o = index_to_position(out_index, out_strides) # get the 1d position of out        |
                j = index_to_position(in_index, in_strides) # get the 1d position of in           |
                out[o] = fn(in_storage[j])                                                        |
        '''                                                                                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/oliviamei/cornell_stuff/mle_codebase/mod3-olivesmoo/minitorch/fast_ops.py
 (226)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/oliviamei/cornell_stuff/mle_codebase/mod3-olivesmoo/minitorch/fast_ops.py (226)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                                              |
        out: Storage,                                                                                                                                                      |
        out_shape: Shape,                                                                                                                                                  |
        out_strides: Strides,                                                                                                                                              |
        a_storage: Storage,                                                                                                                                                |
        a_shape: Shape,                                                                                                                                                    |
        a_strides: Strides,                                                                                                                                                |
        b_storage: Storage,                                                                                                                                                |
        b_shape: Shape,                                                                                                                                                    |
        b_strides: Strides,                                                                                                                                                |
    ) -> None:                                                                                                                                                             |
        # TODO: Implement for Task 3.1.                                                                                                                                    |
        if np.array_equal(a_strides, b_strides) and np.array_equal(a_shape, b_shape) and np.array_equal(a_shape, out_shape) and np.array_equal(a_strides, out_strides):    |
            for i in prange(len(out)):-------------------------------------------------------------------------------------------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                                                                                                    |
        else:                                                                                                                                                              |
            a_indices = np.empty((len(out), len(a_shape)), dtype=np.int32)                                                                                                 |
            b_indices = np.empty((len(out), len(b_shape)), dtype=np.int32)                                                                                                 |
            out_indices = np.empty((len(out), len(out_shape)), dtype=np.int32)                                                                                             |
            for i in prange(len(out)):-------------------------------------------------------------------------------------------------------------------------------------| #3
                to_index(i, out_shape, out_indices[i])                                                                                                                     |
                o = index_to_position(out_indices[i], out_strides)                                                                                                         |
                broadcast_index(out_indices[i], out_shape, a_shape, a_indices[i])                                                                                          |
                j = index_to_position(a_indices[i], a_strides)                                                                                                             |
                broadcast_index(out_indices[i], out_shape, b_shape, b_indices[i])                                                                                          |
                k = index_to_position(b_indices[i], b_strides)                                                                                                             |
                out[o] = fn(a_storage[j], b_storage[k])                                                                                                                    |
            # print(out)                                                                                                                                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/oliviamei/cornell_stuff/mle_codebase/mod3-olivesmoo/minitorch/fast_ops.py
 (293)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/oliviamei/cornell_stuff/mle_codebase/mod3-olivesmoo/minitorch/fast_ops.py (293)
----------------------------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                                                      |
        out: Storage,                                                                                                 |
        out_shape: Shape,                                                                                             |
        out_strides: Strides,                                                                                         |
        a_storage: Storage,                                                                                           |
        a_shape: Shape,                                                                                               |
        a_strides: Strides,                                                                                           |
        reduce_dim: int,                                                                                              |
    ) -> None:                                                                                                        |
        # TODO: Implement for Task 3.1.                                                                               |
        out_indices = np.empty((len(out), len(out_shape)), dtype=np.int32)                                            |
        reduce_size = a_shape[reduce_dim]                                                                             |
                                                                                                                      |
        for i in prange(len(out)):------------------------------------------------------------------------------------| #4
            to_index(i, out_shape, out_indices[i])                                                                    |
            o = index_to_position(out_indices[i], out_strides) # have the 1d index of out                             |
            j = index_to_position(out_indices[i], a_strides) # get first position of the dim we want to accumulate    |
            c = a_strides[reduce_dim] # difference between indices of reduce dim                                      |
            for _ in range(reduce_size):                                                                              |
                out[i] = fn(out[o], a_storage[j])                                                                     |
                j += c                                                                                                |
                                                                                                                      |
        '''                                                                                                           |
        out_index: Index = np.zeros(MAX_DIMS, np.int32)                                                               |
        reduce_size = a_shape[reduce_dim]                                                                             |
        for i in range(len(out)):                                                                                     |
            to_index(i, out_shape, out_index)                                                                         |
            o = index_to_position(out_index, out_strides)                                                             |
            for s in range(reduce_size):                                                                              |
                out_index[reduce_dim] = s                                                                             |
                j = index_to_position(out_index, a_strides)                                                           |
                out[o] = fn(out[o], a_storage[j])                                                                     |
        '''                                                                                                           |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/oliviamei/cornell_stuff/mle_codebase/mod3-olivesmoo/minitorch/fast_ops.py
 (330)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/oliviamei/cornell_stuff/mle_codebase/mod3-olivesmoo/minitorch/fast_ops.py (330)
----------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                |
    out: Storage,                                                           |
    out_shape: Shape,                                                       |
    out_strides: Strides,                                                   |
    a_storage: Storage,                                                     |
    a_shape: Shape,                                                         |
    a_strides: Strides,                                                     |
    b_storage: Storage,                                                     |
    b_shape: Shape,                                                         |
    b_strides: Strides,                                                     |
) -> None:                                                                  |
    """NUMBA tensor matrix multiply function.                               |
                                                                            |
    Should work for any tensor shapes that broadcast as long as             |
                                                                            |
    ```                                                                     |
    assert a_shape[-1] == b_shape[-2]                                       |
    ```                                                                     |
                                                                            |
    Optimizations:                                                          |
                                                                            |
    * Outer loop in parallel                                                |
    * No index buffers or function calls                                    |
    * Inner loop should have no global writes, 1 multiply.                  |
                                                                            |
                                                                            |
    Args:                                                                   |
    ----                                                                    |
        out (Storage): storage for `out` tensor                             |
        out_shape (Shape): shape for `out` tensor                           |
        out_strides (Strides): strides for `out` tensor                     |
        a_storage (Storage): storage for `a` tensor                         |
        a_shape (Shape): shape for `a` tensor                               |
        a_strides (Strides): strides for `a` tensor                         |
        b_storage (Storage): storage for `b` tensor                         |
        b_shape (Shape): shape for `b` tensor                               |
        b_strides (Strides): strides for `b` tensor                         |
                                                                            |
    Returns:                                                                |
    -------                                                                 |
        None : Fills in `out`                                               |
                                                                            |
    """                                                                     |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                  |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                  |
                                                                            |
    # TODO: Implement for Task 3.2.                                         |
    reduce_size = a_shape[-1] # should be equal to a_shape[-2]              |
    for i in prange(len(out)):----------------------------------------------| #5
        cur_ord = i + 0                                                     |
        batch_idx = cur_ord // (out_shape[-2] * out_shape[-1])              |
        cur_ord %= (out_shape[-2] * out_shape[-1])                          |
        row_idx = cur_ord // out_shape[-1]                                  |
        col_idx = cur_ord % out_shape[-1]                                   |
                                                                            |
        a_pos = (batch_idx * a_batch_stride) + (row_idx * a_strides[-2])    |
        b_pos = (batch_idx * b_batch_stride) + (col_idx * b_strides[-1])    |
                                                                            |
        #=====================================                              |
        # cur_ord = i + 0                                                   |
        # a_pos = 0                                                         |
        # b_pos = 0                                                         |
                                                                            |
        # batch_idx = cur_ord // (out_shape[-2] * out_shape[-1])            |
        # cur_ord %= (out_shape[-2] * out_shape[-1])                        |
        # row_idx = cur_ord // out_shape[-1]                                |
        # col_idx = cur_ord % out_shape[-1]                                 |
                                                                            |
        # # a_pos = (batch_idx * a_batch_stride)                            |
        # # b_pos = (batch_idx * b_batch_stride)                            |
                                                                            |
        # for j in range(len(out_shape) - 3, -1, -1):                       |
        #     sh = out_shape[j]                                             |
        #     idx = batch_idx % sh                                          |
        #     if j  == len(a_shape) - 3:                                    |
        #         a_pos += idx * a_batch_stride                             |
        #     elif j < len(a_shape) - 3:                                    |
        #         a_pos += a_strides[j] * idx                               |
                                                                            |
        #     if j == len(b_shape) - 3:                                     |
        #         b_pos += idx * b_batch_stride                             |
        #     elif j < len(b_shape) - 3:                                    |
        #         b_pos += idx * b_strides[j]                               |
        #     batch_idx = batch_idx // sh                                   |
                                                                            |
        # a_pos += row_idx * a_strides[-2]                                  |
        # b_pos += col_idx * b_strides[-1]                                  |
                                                                            |
        a_change = a_strides[-1]                                            |
        b_change = b_strides[-2]                                            |
        sum = 0                                                             |
        for _ in range(reduce_size):                                        |
            sum += a_storage[a_pos] * b_storage[b_pos]                      |
            a_pos += a_change                                               |
            b_pos += b_change                                               |
        out[i] = sum                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

## Module 3.4
| Size  | CPU(s) | GPU(s) |
| ------------- | ------------- | -------------|
| 64  | 0.013601  | 0.011771|
| 128  | 0.038598  | 0.035355
| 256  | 0.205093  | 0.073692
| 512  | 1.100686  | 0.221861
| 1024  | 8.617140  | 1.259826
<img width="662" alt="matrix multiplication times" src="https://github.com/user-attachments/assets/a80eb5da-9fec-4c37-9354-f7a1599343cc">


## Module 3.5
### Simple
#### CPU

```!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --RATE 0.05```

Average time per epoch: 0.14616 seconds

<img width="663" alt="cpu simple 100" src="https://github.com/user-attachments/assets/859f0e2e-ab00-4632-a4c2-c3d47d14c43e">

#### GPU

```!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --RATE 0.05```

Average time per epoch: 2.00573 seconds

<img width="627" alt="gpu simple 100" src="https://github.com/user-attachments/assets/ff5f4012-4dc6-4b60-abca-0fde7fdb063e">

### Simple - 500 hidden
#### CPU

```!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --RATE 0.05```

Average time per epoch: 0.86628 seconds

<img width="635" alt="cpu simple 500" src="https://github.com/user-attachments/assets/faa3fa65-4930-48ca-b2a9-8b506611eb76">


#### GPU

```!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 500 --RATE 0.05```

Average time per epoch: 2.60810 seconds

<img width="642" alt="simple gpu 500" src="https://github.com/user-attachments/assets/eadc9c1c-7040-4847-905f-ed014a365f08">


### Split
#### CPU

```!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05```

Average time per epoch: 0.14599 seconds

<img width="640" alt="cpu split 100" src="https://github.com/user-attachments/assets/4f3d890c-c7eb-4225-bbfe-131858085b10">

#### GPU

```!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05```

Average time per epoch: 1.98883 seconds

<img width="623" alt="gpu split 100" src="https://github.com/user-attachments/assets/a09db2ac-6667-483e-8a1a-478dc31cd486">

### XOR
#### CPU

```!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05```

Average time per epoch: 0.14254 seconds

<img width="631" alt="cpu xor 100" src="https://github.com/user-attachments/assets/f3c9eedd-56e3-43f2-bde5-b563e861c2f3">

#### GPU

```!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05```

Average time per epoch: 1.96577 seconds

<img width="622" alt="gpu xor 100" src="https://github.com/user-attachments/assets/f47c8bac-9939-42d2-86a4-b9b1cde00662">





