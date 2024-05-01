# Copyright (c) 2021 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper

from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim #DD not used?
from qonnx.util.basic import qonnx_make_model


class QuantAvgPool3d(CustomOp):
    """CustomOp that corresponds to the quantized average pooling
    layer from Brevitas"""

    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel": ("i", True, 1),
            "ibits": ("i", True, 1),
            "obits": ("i", True, 1),
            # determines if values are signed (set to "1") or unsigned ("0")
            "signed": ("i", True, 0, {0, 1}),
            "data_layout": ("s", False, "NCDHW"),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        k = self.get_nodeattr("kernel")
        s = self.get_nodeattr("stride")
        data_layout = self.get_nodeattr("data_layout")
        if data_layout == "NCDHW":
            return helper.make_node(
                "AveragePool",
                inputs=[node.input[0]],
                outputs=[node.output[0]],
                kernel_shape=[k, k, k],
                strides=[s, s, s],
            )
        else:
            raise Exception(
                """Datalayout for QuantAvgPool3d is set to an invalid value.
                    Has to be set to "NCDHW"."""
            )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        bw = self.get_nodeattr("obits")
        if bw in range(2, 33):
            if self.get_nodeattr("signed") == 0:
                dtype = DataType["UINT%d" % bw]
            else:
                dtype = DataType["INT%d" % bw]
        else:
            raise Exception("Unsupported output datatype for QuantAvgPool3d")
        model.set_tensor_datatype(node.output[0], dtype)

    def get_accum_size(self):
        ibits = self.get_nodeattr("ibits")
        k = self.get_nodeattr("kernel")
        max_value = 2**ibits - 1
        max_value = max_value * k * k
        max_bit_width = int(max_value).bit_length()
        return max_bit_width

    def get_shifts(self):
        shift_bits = self.get_accum_size() - self.get_nodeattr("obits")
        shift_bits = shift_bits if shift_bits >= 0 else 0
        return shift_bits

    def execute_node(self, context, graph):
        # create a standard average pooling node to help calculate the result
        node = self.onnx_node
        k = self.get_nodeattr("kernel")
        s = self.get_nodeattr("stride")
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        node_avgpool = helper.make_node(
            "AveragePool",
            inputs=[node.input[0]],
            outputs=[node.output[0]],
            kernel_shape=[k, k, k],
            strides=[s, s, s],
        )
        graph_avgpool = helper.make_graph(
            nodes=[node_avgpool],
            name="single-avgpool-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_avgpool = qonnx_make_model(graph_avgpool, **onnx_kwargs)
        idict = {node.input[0]: inp_values}
        sess = rt.InferenceSession(model_avgpool.SerializeToString())
        result_temp = sess.run(None, idict)
        # remove scaling introduced by average
        result_temp = np.round(result_temp[0] * (k * k * k))
        result = np.right_shift(result_temp.astype(int), self.get_shifts())
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        info_messages = []
        return info_messages
