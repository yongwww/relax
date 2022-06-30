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
# under the License.
# pylint: disable=unused-argument, invalid-name, no-else-return
"""Relay to Relax translator."""

from __future__ import annotations
from ast import Nonlocal
from ctypes import Union
from turtle import clear
from typing import Any, Dict, List, Optional
import tvm
from tvm.ir.module import IRModule
from tvm import relax, relay
from tvm.relax.testing import nn
from tvm.relay.backend.te_compiler import select_implementation
from tvm.runtime import NDArray
from tvm.target import Target
from tvm.meta_schedule.utils import autotvm_silencer


def from_relay(
    relay_model: Union[relay.Function, tvm.ir.IRModule],
    target: Target,
    relay_params: Optional[Dict[str, NDArray]] = None,
    *,
    opt_level: int = 3,
    pass_config: Optional[Dict[str, Any]] = None,
    disabled_pass: Optional[List[str]] = None,
    translate_op_with_tir: Optional[Dict[str, tvm.tir.PrimFunc]] = None,
) -> IRModule:
    """Convert a Relay function into a Relax program.

    Parameters
    ----------
    relay_model : Union[relay.Function, tvm.ir.IRModule]
        Relay function or TVM IRModule to be converted.

    target: Target
        The target to compile the model, used for selecting topi functions.

    relay_params: Optional[Dict[str, NDArray]]
        Parameters to bind.

    opt_level: int
        The optimization level.

    pass_config: Optional[Dict[str, Any]]
        Pass configuration.

    disabled_pass: Optional[List[str]]
        Passes to disable.

    translate_op_with_tir: Optional[Dict[str, tvm.tir.PrimFunc]]
        Dict that maps op names to user-defined PrimFuncs.
        Takes relay operator names and forces them to user-defined PrimFuncs during translation.

    Returns
    -------
    mod : tvm.IRModule
        The Relax IRModule for compilation
    """
    # A map to store the mapping of Relay Expr to its corresponding Relax var
    var_map = {}
    # The manually creted relax local func, it will be used to emit for relay.Let var = fn()
    local_func = None
    # var for the local func, which will be used in the recursive call, and call outside local func
    local_func_var = relax.Var(
        "local_func",
        # shape_annotation #
        type_annotation=relax.ty.TupleType(
            [
                relax.ty.DynTensorType(0, "int32"),
                # todo (yongwww): present List[DynTensorType]
                relax.ObjectType(),
                relax.ty.TupleType(
                    [
                        relax.ty.DynTensorType(2, "float32"),
                        relax.ty.DynTensorType(2, "float32"),
                    ]
                ),
                relax.ty.DynTensorType(3, "float32"),
            ]
        ),
    )

    # The output of the function
    output_var = None
    # running vars
    params = []
    # params for the entry function
    main_params = []
    # params for local function
    local_func_params = []
    # node counter
    counter = 0

    if not isinstance(target, Target):
        target = Target(target)
    if disabled_pass is None:
        disabled_pass = []
    if pass_config is None:
        pass_config = {
            "relay.FuseOps.max_depth": 1,  # Disable relay fusion
            "relay.backend.use_meta_schedule": True,
        }

    if relay_params:
        func = relay.build_module.bind_params_by_name(func, relay_params)

    def convert_shape(shape: List[tvm.tir.PrimExpr]) -> List[tvm.tir.PrimExpr]:
        """Convert the relay shape to relax shape by changing Any dim to symbolic dim"""
        ret = []
        for dim in shape:
            if isinstance(dim, tvm.tir.IntImm):
                ret.append(tvm.tir.IntImm("int64", int(dim)))
            elif isinstance(dim, tvm.tir.Any):
                ret.append(tvm.tir.Var("d", "int64"))
            else:
                ret.append(dim)
        return ret

    true_branch = None
    false_branch = None

    def visit_func(node):
        nonlocal counter
        nonlocal params
        nonlocal true_branch
        nonlocal false_branch
        nonlocal main_params
        nonlocal local_func
        nonlocal local_func_var
        nonlocal local_func_params
        nonlocal output_var

        counter = counter + 1
        print("translating the {}th node with type {}".format(counter, type(node)))

        if counter == 14:  # %eta_expand_param
            main_params = params
        elif counter == 17:  # eta function
            params = []
        elif counter == 22:  # constant 7
            for param in params:
                local_func_params.append(param)
        elif counter == 23:  # constant 7
            bb._begin_binding_block()
        elif counter == 25:  # less
            bb._begin_binding_block()
        elif counter == 100:
            blocks = [bb._end_block()]
            true_branch = relax.SeqExpr(blocks, blocks[-1].bindings[-1].var)
            bb._begin_binding_block()

        if isinstance(node, relay.Var):
            if isinstance(node.type_annotation, relay.TensorType):
                var_map[node] = nn.Placeholder(
                    tuple(convert_shape(node.type_annotation.shape)),
                    node.type_annotation.dtype,
                    node.name_hint,
                )
                params.append(var_map[node])
            elif isinstance(node.type_annotation, relay.TupleType):
                rx_shape = []
                rx_type = []
                for f in node.type_annotation.fields:
                    field_shape = relax.ShapeExpr(convert_shape(f.shape))
                    rx_shape.append(field_shape)
                    rx_type.append(relax.DynTensorType(ndim=len(field_shape.values), dtype=f.dtype))

                rx_tuple_shape = relax.Tuple(rx_shape)
                rx_tuple_type = relax.TupleType(rx_type)

                v = relax.Var(
                    node.name_hint,
                    shape_annotation=rx_tuple_shape,
                    type_annotation=rx_tuple_type,
                )
                var_map[node] = v
                params.append(var_map[node])
            elif isinstance(node.type_annotation, tvm.ir.type_relation.TypeCall):
                #  %outputs.11: List[Tensor[(1, 4), float32]] /* ty=List[Tensor[(1, 4), float32]] */;
                v = relax.Var(node.name_hint, type_annotation=relax.ObjectType())
                var_map[node] = v
                params.append(var_map[node])

            elif isinstance(node.type_annotation, tvm.ir.type.FuncType):
                v = relax.Var(node.name_hint, type_annotation=relax.ObjectType())
                var_map[node] = local_func_var
                params.append(var_map[node])

            else:
                raise TypeError("The type of relay.Var to be translated must be of TensorType.")
        elif isinstance(node, relay.Call):
            args = node.args
            new_args = []
            te_inputs = []
            if "Cons" in str(node.op) and isinstance(node.op, tvm.ir.adt.Constructor):
                tl = var_map[args[1]]
                write_idx = relax.const(0)
                write_val = var_map[args[0]]
                call = relax.Call(
                    relax.ExternFunc("relax.run.tensor_list_write"), (tl, write_idx, write_val)
                )
                var = bb.emit(call)
                output_var = var
                var_map[node] = var
                return

            if "@concat" in str(node.op) and isinstance(node.op, relay.GlobalVar):
                call = relax.Call(
                    relax.ExternFunc("relax.run.concat_lists"),
                    (var_map[args[0]], var_map[args[1]]),
                )
                var = bb.emit(call)
                output_var = var
                var_map[node] = var
                return

            if isinstance(node.op, relay.expr.Let):
                # %40 = %39(0, %38, %states, %input); # let
                call = relax.Call(
                    local_func_var,  # var_map[node.op],
                    [var_map[args[0]], var_map[args[1]], var_map[args[2]], var_map[args[3]]],
                )
                var = bb.emit(call)
                output_var = var
                var_map[node] = var
                return
            if "%while_loop" in str(node.op) and isinstance(node.op, relay.Var):
                # %while_loop(%35, %36, %37, %input.1)
                rec_call_op = relax.GlobalVar(
                    "local_func",
                    relax.ty.TupleType(
                        [
                            relax.ty.DynTensorType(0, "int32"),
                            # todo (yongwww): present List[DynTensorType]
                            relax.ObjectType(),
                            relax.ty.TupleType(
                                [
                                    relax.ty.DynTensorType(2, "float32"),
                                    relax.ty.DynTensorType(2, "float32"),
                                ]
                            ),
                            relax.ty.DynTensorType(3, "float32"),
                        ]
                    ),
                )
                call = relax.Call(
                    rec_call_op,  # local_func_var,  # var_map[node.op],
                    [var_map[args[0]], var_map[args[1]], var_map[args[2]], var_map[args[3]]],
                )

                # Need to manually update shape and type for the call
                # since the local funcion has not been created yet
                # otherwise will run into InferType and InferShape issue
                # todo (yongwww): update the related shape and type for each element of tuple

                # %while_loop(%35, %36, %37, %input.1)
                # /* ty=(int32,
                #        List[Tensor[(1, 4), float32]],
                #        (Tensor[(1, 4), float32], Tensor[(1, 4), float32]),
                #        Tensor[(7, 1, 3), float32]) */
                rx_shape = relax.Tuple(
                    [
                        relax.ShapeExpr([]),
                        relax.RuntimeDepShape(),  # todo (yongwww): List[(1, 4)]
                        relax.Tuple([relax.ShapeExpr([1, 4]), relax.ShapeExpr([1, 4])]),
                        relax.ShapeExpr([7, 1, 3]),
                    ]
                )
                rx_type = relax.ty.TupleType(
                    [
                        relax.ty.DynTensorType(0, "int32"),
                        # todo (yongwww): present List[DynTensorType]
                        relax.ObjectType(),
                        relax.ty.TupleType(
                            [
                                relax.ty.DynTensorType(2, "float32"),
                                relax.ty.DynTensorType(2, "float32"),
                            ]
                        ),
                        relax.ty.DynTensorType(3, "float32"),
                    ]
                )
                relax.expr._update_shape(call, rx_shape)
                relax.expr._update_type(call, rx_type)

                var = bb.emit(call)
                output_var = var
                var_map[node] = var
                return

            if isinstance(node.op, relay.GlobalVar):
                if "tensor_array_stack" in str(node.op.name_hint):
                    call = relax.Call(
                        relax.ExternFunc("relax.run.tensor_list_stack"), ([var_map[args[0]]])
                    )  # todo(yongwww): params
                elif "tensor_get_data" in str(node.op.name_hint):
                    call = relax.Call(
                        relax.ExternFunc("relax.run.tensor_list_read"),
                        (var_map[args[0]], relax.const(0)),
                    )
                elif "map" in str(node.op.name_hint):
                    call = relax.Call(relax.ExternFunc("vm.builtin.empty_list"), [])
                    var_lst = bb.emit(call)
                    call = relax.Call(
                        relax.ExternFunc("relax.run.tensor_list_write"),
                        (var_lst, relax.const(0), var_map[args[1]]),
                    )
                var = bb.emit(call)
                output_var = var
                var_map[node] = var
                return

            for arg in args:
                if arg in var_map:
                    new_args.append(var_map[arg])
                    te_inputs.append(relax.expr.te_tensor(new_args[-1]))

            if isinstance(node.op, tvm.ir.adt.Constructor):
                # tensor_constructor_float32_1_4(Tensor[(1, 4), float32])
                call = relax.Call(relax.ExternFunc("vm.builtin.empty_list"), [])
                var = bb.emit(call)
                # var.checked_type should be objectType
                output_var = var
                var_map[node] = var
                return

            op_name = node.op.name
            # if map, return output_var
            attrs = node.attrs
            out_type = node.checked_type

            if translate_op_with_tir and op_name in translate_op_with_tir:
                tir_gvar = bb.add_func(translate_op_with_tir[op_name], op_name)
                call = relax.call_tir(tir_gvar, new_args, out_type.shape, out_type.dtype)
                var = bb.emit(call)
            else:
                best_impl, outputs = select_implementation(
                    node.op,
                    attrs,
                    te_inputs,
                    out_type,
                    target,
                    use_autotvm=False,
                )
                compute_func = best_impl.compute
                name_hint = op_name.split(".")[-1]
                var = bb.emit_te(
                    compute_func, attrs, new_args, node.checked_type, primfunc_name_hint=name_hint
                )

            output_var = var
            var_map[node] = var
        elif isinstance(node, relay.Constant):
            # fill the shape and checked_type fields of the Constant
            new_constant = relay.Constant(node.data)
            var_map[node] = new_constant
        elif isinstance(node, relay.Tuple):
            new_fields = []
            for field in node.fields:
                if field in var_map:
                    new_fields.append(var_map[field])
                else:
                    raise RuntimeError("field is not in var_map.")
            new_tuple = relax.Tuple(new_fields)
            new_tuple_var = relax.BlockBuilder.current().emit(new_tuple)
            var_map[node] = new_tuple_var
            if counter == 100:
                blocks = [bb._end_block()]
                false_branch = relax.SeqExpr(blocks, blocks[-1].bindings[0].var)

            output_var = new_tuple_var
        elif isinstance(node, relay.TupleGetItem):
            if node.tuple_value in var_map:
                new_tuple = var_map[node.tuple_value]
                # print("new_tuple: {} \n new_tuple type : {}".format(new_tuple, type(new_tuple)))
                new_tuple_get_item_node = relax.TupleGetItem(new_tuple, node.index)
                new_tuple_get_item_var = relax.BlockBuilder.current().emit(new_tuple_get_item_node)
                var_map[node] = new_tuple_get_item_var
                output_var = new_tuple_get_item_var
            else:
                raise RuntimeError("tuple is not in var_map")
        elif isinstance(node, relay.Function):
            # eta_expand_params skip.
            if node.params[0].name_hint == "eta_expand_param":
                return
            # local func: manually create a func, relax.function (body: seqexpr,)
            elif node.params[0].name_hint == "i.1":
                blocks = [bb._end_block()]
                body = relax.SeqExpr(blocks, blocks[-1].bindings[-1].var)
                # %while_loop(%35, %36, %37, %input.1)
                # /* ty=(int32,
                #        List[Tensor[(1, 4), float32]],
                #        (Tensor[(1, 4), float32], Tensor[(1, 4), float32]),
                #        Tensor[(7, 1, 3), float32]) */

                ret_type = relax.ty.TupleType(
                    [
                        relax.ty.DynTensorType(0, "int32"),
                        # todo (yongwww): present List[DynTensorType]
                        relax.ObjectType(),
                        relax.ty.TupleType(
                            [
                                relax.ty.DynTensorType(2, "float32"),
                                relax.ty.DynTensorType(2, "float32"),
                            ]
                        ),
                        relax.ty.DynTensorType(3, "float32"),
                    ]
                )
                local_func = relax.Function(local_func_params, body, ret_type)
                return

            cur_bb = relax.BlockBuilder.current()
            cur_bb.emit_func_output(output_var, main_params)
        elif isinstance(node, tvm.ir.Op):
            pass
        elif isinstance(node, relay.GlobalVar):
            # @tensor_get_data_float32_any_1_4
            pass
        elif isinstance(node, tvm.ir.adt.Constructor):
            pass
        elif isinstance(node, relay.expr.If):
            # node.cond, node.true_branch, node.false_branch
            rx_if = relax.If(var_map[node.cond], true_branch, false_branch)
            var_if = bb.emit(rx_if)
            var_map[node] = var_if
        elif isinstance(node, relay.expr.Let):
            # let %while_loop = fn (%i.1:...
            var_bind = relax.VarBinding(local_func_var, local_func)
            var_fn = bb.emit_varbinding(var_bind)
            # var_fn = bb.emit(local_func)
            var_map[node] = var_fn
        else:
            raise TypeError("{} is not supported yet.".format(str(type(node))))

    # List of subset of relay->relay optimizations
    # See src/relay/backend/utils.cc::GetPassPrefix() for full list
    seq = tvm.get_global_func("relay.backend.GetPassPrefixSeq")(True, True)

    # Since optimization passes and OpStrategy are highly context-dependent,
    # we match the exact same context with `extract_task_from_relay()` env
    with autotvm_silencer(), target, tvm.transform.PassContext(
        opt_level=opt_level, config=pass_config, disabled_pass=disabled_pass
    ):
        if isinstance(relay_model, tvm.ir.IRModule):
            mod = relay_model
        else:
            mod = tvm.IRModule.from_expr(relay_model)

        mod = seq(mod)

        bb = relax.BlockBuilder()

        with bb.function("main"):
            bb._begin_binding_block()
            relay.analysis.post_order_visit(mod["main"], visit_func)
    # print("relax lstm IR: \n", bb.get()["main"])
    return bb.get()
