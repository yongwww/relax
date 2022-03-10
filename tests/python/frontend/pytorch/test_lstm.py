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
""" Tests on torch lstm model conversion """
# originally from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
# described in https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor

import tvm
import tvm.testing
from tvm import relay
from tvm.relay.frontend.pytorch import from_pytorch
from tvm.relay.prelude import Prelude
from tvm.runtime.container import ADT, tuple_object


class LayerNormLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))

        ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        outputs = []
        # print("input: ", input)
        for i in range(input.size(0)):
            out, state = self.cell(input[i], state)
            outputs += [out]
        return torch.stack(outputs), state


def lstm(input_size, hidden_size):
    print("yongwww, input_size: {}, hidden_size: {}".format(input_size, hidden_size))
    return LSTMLayer(LayerNormLSTMCell, input_size, hidden_size)
    # lstm(input_size, hidden_size).eval()
    """
    input_name = "input"
    states_name = "states"
    seq_len = 7
    batch = 1
    input_size = 3
    hidden_size = 4
    num_layers = 3
    """


def stacked_lstm(input_size, hidden_size, num_layers):
    return StackedLSTM(
        num_layers,
        LSTMLayer,
        first_layer_args=[LayerNormLSTMCell, input_size, hidden_size],
        other_layer_args=[LayerNormLSTMCell, hidden_size, hidden_size],
    )


def bidir_lstm(input_size, hidden_size):
    return BidirLSTMLayer(LayerNormLSTMCell, input_size, hidden_size)


def stacked_bidir_lstm(input_size, hidden_size, num_layers):
    return StackedBidirLSTM(
        num_layers,
        BidirLSTMLayer,
        first_layer_args=[LayerNormLSTMCell, input_size, hidden_size],
        other_layer_args=[LayerNormLSTMCell, hidden_size, hidden_size],
    )


def vmobj_to_list(o, dtype="float32"):
    if isinstance(o, tvm.nd.NDArray):
        return [o]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f, dtype))
        return result
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def assert_equal(tvm_result, torch_result):
    if isinstance(torch_result, (tuple, list)):
        assert isinstance(tvm_result, list)
        for tvm_res, pt_res in zip(tvm_result, torch_result):
            assert_equal(tvm_res, pt_res)
    elif isinstance(torch_result, torch.Tensor):
        tvm.testing.assert_allclose(tvm_result.numpy(), torch_result.numpy(), rtol=1e-4, atol=1e-4)


def run_and_compare(mod, params, pt_result, target, device):
    exec_res = relay.create_executor("vm", mod=mod, device=device, target=target).evaluate()(
        **params
    )

    def flatten(nested):
        res = []
        for r in nested:
            if isinstance(r, torch.Tensor):
                res.append(r)
            else:
                res.extend(flatten(r))
        return res

    if isinstance(exec_res, tvm.runtime.container.ADT):
        assert not isinstance(pt_result, torch.Tensor)
        tvm_res = vmobj_to_list(exec_res)
        torch_res = flatten(pt_result)
    else:
        tvm_res = exec_res
        torch_res = pt_result

    assert_equal(tvm_res, torch_res)


def convert_list_to_vmobj(py_lst):
    def wrap_nd_array(arr):
        return tvm.nd.array(arr, device=tvm.cpu(0))

    mod = tvm.IRModule()
    prelude = Prelude(mod)
    list, cons, nil = mod.get_type("List")
    adt_lst = ADT(nil.tag, [])
    for elem in reversed(py_lst):
        if isinstance(elem, np.ndarray):
            vmobj = wrap_nd_array(elem)
        elif isinstance(elem, tuple):
            vmobj = tuple_object([wrap_nd_array(e) for e in elem])
        elif isinstance(elem, list):
            vmobj = convert_list_to_vmobj(elem)
        adt_lst = ADT(cons.tag, [vmobj, adt_lst])
    return adt_lst


@tvm.testing.uses_gpu
def test_custom_lstm():
    input_name = "input"
    states_name = "states"
    seq_len = 7
    batch = 1
    input_size = 3
    hidden_size = 4
    num_layers = 3
    state_tensor_shape = (batch, hidden_size)

    torch.manual_seed(1)

    inp = torch.randn(seq_len, batch, input_size)

    input_shapes = [
        (input_name, (seq_len, batch, input_size)),
        (states_name, (state_tensor_shape, state_tensor_shape)),
    ]

    states = [
        (torch.randn(state_tensor_shape), torch.randn(state_tensor_shape))
        for _ in range(num_layers)
    ]

    models = [
        ("lstm", lstm(input_size, hidden_size).eval(), states[0], input_shapes),
        """
        (
            "stacked",
            stacked_lstm(input_size, hidden_size, num_layers).eval(),
            states,
            input_shapes_stacked,
        ),
        ("bidir", bidir_lstm(input_size, hidden_size).eval(), bidir_states, input_shapes_stacked),
        """
        # TODO(masahi): stacked bidir seems to have a rare accuracy issue
        # (
        #     "stacked_bidir",
        #     stacked_bidir_lstm(input_size, hidden_size, num_layers).eval(),
        #     stacked_bidir_states,
        #     input_shapes_stacked_bidir,
        # ),
    ]  # todo

    for (name, raw_model, states, input_shapes) in models:
        script_module = torch.jit.script(raw_model)
        mod, params = from_pytorch(script_module, input_shapes)
        print("relay lstm mod: ", mod)

        with torch.no_grad():
            pt_result = raw_model(inp.clone(), states)

        params[input_name] = inp.numpy()

        if isinstance(states, tuple):
            states_np = tuple(st.numpy() for st in states)
        elif isinstance(states, list) and isinstance(states[0], torch.Tensor):
            states_np = [st.numpy() for st in states]
        elif isinstance(states, list) and isinstance(states[0], tuple):
            states_np = [tuple(st.numpy() for st in states[i]) for i in range(len(states))]
        elif isinstance(states, list) and isinstance(states[0], list):
            states_np = [
                [tuple(st.numpy() for st in states) for states in states[layer]]
                for layer in range(num_layers)
            ]
        else:
            assert False

        if isinstance(states_np, list):
            params[states_name] = convert_list_to_vmobj(states_np)
        else:
            params[states_name] = states_np

        for tgt, dev in tvm.testing.enabled_targets():
            print("Running %s on target %s" % (name, tgt))
            run_and_compare(mod, params, pt_result, target=tgt, device=dev)


if __name__ == "__main__":
    test_custom_lstm()
