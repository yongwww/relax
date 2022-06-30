# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
"""Basic tensor operations."""
import numpy as np
import tensorflow as tf
import tvm

from . import _ffi_api
from typing import List
from ..expr import Expr


@tvm.register_func("relax.run.tensor_list_stack")
def tensor_list_stack(
    data: List[tvm.nd.array],  #
) -> tvm.nd.array:
    """Returns the stacked tensor.
    Uses TensorArray.stack() of TensorFlow to compute.
    """
    data = list(data)
    dtype = tf.float32  # todo (@yongwww): fix the hardcoded datatype
    # TODO(@yongwww): support for tensor with different size
    # Construct TensorFlow TensorArray
    ta = tf.TensorArray(dtype=dtype, size=len(data), infer_shape=True)
    for i, d in enumerate(data):
        ta.write(
            i, tf.experimental.dlpack.from_dlpack(d.to_dlpack())
        )  # via dltensor (TVM NDArray to tf Tensor)

    # Perform Stack to get a TensorFlow Tensor
    tf_out = ta.stack()
    # tf.make_ndarray(output) failed with "EagerTensor' object
    # has no attribute 'tensor_shape'"

    # Convert tf Tensor to (numpy then to) NDArray
    np_out = tf_out.numpy()  # tf.make_ndarray(tf_out)

    return tvm.nd.array(np_out)


@tvm.register_func("relax.run.tensor_list_write")
def tensor_list_write(
    data: List[tvm.nd.array],
    index: int,
    value: tvm.nd.array,
) -> List[tvm.nd.array]:
    """TensorListWrite"""
    data = list(data)
    dtype = tf.float32  # todo
    # Construct TensorFlow TensorArray
    ta = tf.TensorArray(dtype=dtype, size=len(data) + 1, infer_shape=True)
    for i, d in enumerate(data):
        ta.write(i, tf.experimental.dlpack.from_dlpack(d.to_dlpack()))  # d or d.numpy()

    # Perform write

    idx = index.numpy().item()
    if isinstance(value, tvm.ir.container.Array):
        for val in value:
            ta = ta.write(idx, tf.experimental.dlpack.from_dlpack(val.to_dlpack()))
            # idx = idx + 1 fix
    else:
        ta = ta.write(idx, tf.experimental.dlpack.from_dlpack(value.to_dlpack()))

    output = []
    for i in range(ta.size()):
        tf_out = ta.read(i)
        # todo: tf Tensor to tvm ndarray
        output.append(tvm.nd.array(tf_out.numpy()))
    return output


@tvm.register_func("relax.run.tensor_list_read")
def tensor_list_read(
    data: List[tvm.nd.array],
    index: int,
) -> tvm.nd.array:
    """TensorListRead"""
    if isinstance(data, tvm.nd.NDArray):
        return data  # todo(remove the check)
    data = list(data)
    dtype = tf.float32  # todo
    return data[index.numpy().item()]
    """
    # Construct TensorFlow TensorArray
    ta = tf.TensorArray(dtype=dtype, size=len(data), infer_shape=True)
    for i, d in enumerate(data):
        ta.write(i, tf.experimental.dlpack.from_dlpack(d.to_dlpack()))  # d or d.numpy()

    # Perform TensorArrayRead
    tf_out = ta.read(index.numpy().item())
    # tf.make_ndarray(output) failed with "EagerTensor' object
    # has no attribute 'tensor_shape'"
    np_out = tf_out.numpy()
    out = tvm.nd.array(np_out)
    return out
    """


@tvm.register_func("relax.run.concat_lists")
def concat_lists(
    lhs_list: List[tvm.nd.array],
    rhs_list: List[tvm.nd.array],
) -> List[tvm.nd.array]:
    output = list(lhs_list)
    rhs_list = list(rhs_list)
    for data in rhs_list:
        output.append(data)
    return output
