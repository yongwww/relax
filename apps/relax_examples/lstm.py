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
""" Tests LSTM conversion: Torch -> Relay -> Relax """

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor

import tvm
import tvm.testing
from tvm import relay, relax, transform
from tvm.runtime import vm as vm_rt
from tvm.script import tir as T, relax as R
from tvm.relay.frontend.pytorch import from_pytorch

from tvm.relax.testing import relay_translator
from tvm.relax.testing import nn as _nn
import time


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
        for i in range(input.size(0)):
            out, state = self.cell(input[i], state)
            outputs += [out]
        return torch.stack(outputs), state


def lstm(input_size, hidden_size):
    return LSTMLayer(LayerNormLSTMCell, input_size, hidden_size)


def get_relay_lstm():
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

    script_module = torch.jit.script(lstm(input_size, hidden_size).eval())
    relay_mod, relay_params = from_pytorch(script_module, input_shapes)
    return relay_mod, relay_params


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.numpy()]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f))
        return result
    elif isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
        if o.constructor.name_hint == "Cons":
            tl = vmobj_to_list(o.fields[1])
            hd = vmobj_to_list(o.fields[0])
            hd.extend(tl)
            return hd
        elif o.constructor.name_hint == "Nil":
            return []
        elif "tensor_nil" in o.constructor.name_hint:
            return [0]
        elif "tensor" in o.constructor.name_hint:
            return [o.fields[0].numpy()]
        else:
            raise RuntimeError("Unknown object type: %s" % o.constructor.name_hint)
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


if __name__ == "__main__":
    # get relay
    relay_mod, _ = get_relay_lstm()
    relay_mod = relay.transform.InferType()(relay_mod)

    # translate the LSTM model from Relay to Relax
    target = tvm.target.Target("llvm", host="llvm")
    # relax_mod = relay_translator.from_relay(relay_mod, target)
    # relax_mod = relax.transform.LambdaLift()(relax_mod)
    with transform.PassContext(opt_level=3):
        """
        relay_mod = relay.transform.SimplifyInference()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        relay_mod = relay.transform.FuseOps(fuse_opt_level=2)(relay_mod)
        print("relay_mod after fuse: \n", relay_mod["main"])
        """

        relax_mod = relay_translator.from_relay(relay_mod, target=target)
        relax_mod = relax.transform.LambdaLift()(relax_mod)
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        relax_mod = relax.transform.FuseOps()(relax_mod)
        relax_mod = relax.transform.FuseTIR()(relax_mod)
        # relax_mod = relax.transform.LambdaLift()(relax_mod)

    print("relax main: \n", relax_mod["main"])
    print("relax lifted_func_0: \n", relax_mod["lifted_func_0"])
    """
    print("relax lifted_func_0 : \n", relax_mod["lifted_func_0"])

    relax_text = R.parser.astext(relax_mod, show_meta_data=True)
    # codegen
    relax_ex = relax.vm.build(relax_mod, target)
    # print("ex: \n", ex.as_text())
    # create relax vm
    relax_vm = relax.VirtualMachine(relax_ex, tvm.cpu())
    # init weights and run the model on relax vm
    input_params = _nn.init_params(relax_mod)
    # run on relax vm
    relax_out = relax_vm["main"](*input_params)
    relax_res = vmobj_to_list(relax_out)
    # print("relax inference result: \n", relax_res)

    warmup = 5
    iterations = 50

    # warmup relax
    for _ in range(warmup):
        relax_vm["main"](*input_params)

    relax_start_time = time.time()
    for _ in range(iterations):
        relax_vm["main"](*input_params)
    relax_aver_perf = (time.time() - relax_start_time) / iterations

    # run on relay vm
    relay_ex = relay.vm.compile(relay_mod, target)
    relay_vm = vm_rt.VirtualMachine(relay_ex, tvm.cpu())
    relay_out = relay_vm.run(*input_params[:10])  # a hack, need to remove
    relay_res = vmobj_to_list(relay_out)
    # print("relay inference result: \n", relay_res)

    # check correctness by comparing with relay result
    # tvm.testing.assert_allclose(res.numpy(), expected_output.numpy(), rtol=1e-4, atol=1e-4)

    # Collect relay Inference perf
    # warmup relay
    relax.DataflowBlock
    for _ in range(warmup):
        relay_vm.run(*input_params)

    relay_start_time = time.time()
    for _ in range(iterations):
        relay_vm.run(*input_params)
    relay_aver_perf = (time.time() - relay_start_time) / iterations

    print("Relay LSTM latency (second): ", relay_aver_perf)
    print("Relax LSTM latency (second): ", relax_aver_perf)
    """
