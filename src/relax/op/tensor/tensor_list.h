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
 * \file tensor_list.h
 * \brief shape and type deduction for TensorList operators.
 */

#ifndef TVM_RELAX_OP_TENSOR_LIST__H_
#define TVM_RELAX_OP_TENSOR_LIST__H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <vector>

#include "../op_common.h"

namespace tvm {
namespace relax {

Type ReturnVoidType1(const Call& call, DiagnosticContext diag_ctx) { return VoidType(); }

Type InferTypeTensorListStack(const Call& call, DiagnosticContext diag_ctx) {
  auto* input_ty = call->args[0]->checked_type().as<DynTensorTypeNode>();
  return DynTensorType(/*ndim=*/1, input_ty->dtype);
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_LIST__H_
