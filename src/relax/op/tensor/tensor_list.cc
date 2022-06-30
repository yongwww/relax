/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tensor_list.cc
 * \brief TensorList operators.
 */

#include "tensor_list.h"

namespace tvm {
namespace relax {

// tensor_list_stack

// TVM_REGISTER_NODE_TYPE(TensorListStackAttrs);

RELAY_REGISTER_OP("relax.tensor_list_stack")
    .describe(R"code(Stack the input tensors.

- **data** : A list of tensors.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input list of tensors.")
    .set_attr<FInferType>("FInferType", ReturnVoidType1)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.tensor_list_stack");

// Expr TensorListStack(Expr input_handle, Expr element_shape, Expr element_dtype, Expr
// num_elements) {
Expr MakeTensorListStack(Expr data) {
  static const Op& op = Op::Get("relax.tensor_list_stack");
  // return Call(op, {input_handle, element_shape, element_dtype, num_elements}, {}, {});
  return Call(op, {data}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.tensor_list_stack").set_body_typed(MakeTensorListStack);

// tensor_list_write

TVM_REGISTER_NODE_TYPE(TensorListWriteAttrs);

RELAY_REGISTER_OP("relax.tensor_list_write")
    .describe(R"code(Write an tensor into tensor list along given index.

- **list** : A list of tensors.
- **index** : The index to write tensor.
- **data** : A tensor to be added.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("list", "Tensor", "The list of tensors.")
    // .add_argument("index", "Tensor", "The index to add tensor.")
    .add_argument("data", "Tensor", "The tensor to be added.")
    // .set_attrs_type<TensorListWriteAttrs>()
    .set_attr<FInferType>("FInferType", ReturnVoidType1)  // todo (yongwww)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.tensor_list_write");

Expr MakeTensorListWrite(Expr list, Expr index, Expr data) {
  // auto attrs = make_object<TensorListWriteAttrs>();
  // attrs->index = index;
  static const Op& op = Op::Get("relax.tensor_list_write");
  // todo(yongwww): move index into TensorListWriteAttrs and use int instead of Expr for it
  return Call(op, {list, index, data}, {});
}

TVM_REGISTER_GLOBAL("relax.op.tensor_list_write").set_body_typed(MakeTensorListWrite);

// tensor_list_read

// TVM_REGISTER_NODE_TYPE(TensorListReadAttrs);

RELAY_REGISTER_OP("relax.tensor_list_read")
    .describe(R"code(Read an tensor from tensor list.

- **data** : A list of tensors.
- **index** : The index to read tensor.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The list of tensors.")
    .add_argument("index", "Tensor", "The index to add tensor.")
    .set_attr<FInferType>("FInferType", ReturnVoidType1)  // todo (yongwww)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.tensor_list_read");

Expr MakeTensorListRead(Expr data, Expr index) {
  static const Op& op = Op::Get("relax.tensor_list_read");
  return Call(op, {data, index}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.tensor_list_read").set_body_typed(MakeTensorListRead);

// concat_lists

RELAY_REGISTER_OP("relax.concat_lists")
    .describe(R"code(Concatenate two lists.

- **lhs_list** : A list of tensors.
- **rhs_list** : A list of tensors.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("lhs_list", "Tensor", "The list of tensors.")
    .add_argument("rhs_list", "Tensor", "The list of tensors.")
    .set_attr<FInferType>("FInferType", ReturnVoidType1)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.concat_lists");

Expr MakeConcatLists(Expr lhs_list, Expr rhs_list) {
  static const Op& op = Op::Get("relax.concat_lists");
  return Call(op, {lhs_list, rhs_list}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.concat_lists").set_body_typed(MakeConcatLists);

}  // namespace relax
}  // namespace tvm
