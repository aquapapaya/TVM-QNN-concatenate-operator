import numpy as np

import tvm
from tvm import relay

model_path = './yolov3-416-int8_tf2_3_0.tflite'
input_name = 'input_1'
data_type = 'float32'
data_shape = (1, 416, 416, 3)

######################################################################
# Set target
# ----------

target = {'gpu': 'cuda','cpu':'llvm'}
target_host = 'llvm'
fallback_device = tvm.context("llvm")
ctx = tvm.gpu(0)

######################################################################
# Load test image
# -----------------

import cv2
image_path = "2.jpg"
input_size = 416
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

image_data = cv2.resize(original_image, (input_size, input_size))
image_data = image_data / 255.

images_data = []
for i in range(1):
    images_data.append(image_data)
images_data = np.asarray(images_data).astype(np.float32)

######################################################################
# Load a TFLite model
# -------------------

import os
tflite_model_file = os.path.join(model_path)
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Convert the TFLite model into Relay IR
# --------------------------------------

import tvm.relay as relay
dtype_dict = {input_name: data_type}
shape_dict = {input_name: data_shape}

mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict=shape_dict,
                                         dtype_dict=dtype_dict)

######################################################################
# Compile the Relay module
# ------------------------

from tvm.relay.expr_functor import ExprMutator
class ScheduleQNN_init(ExprMutator):
    def __init__(self, device):
        self.device = device
        super().__init__()

    def visit_call(self, expr):
        visit = super().visit_call(expr)
        if expr.op == tvm.relay.op.get("qnn.concatenate"):
            return relay.annotation.on_device(visit, self.device)
        else:
            return visit

func = mod["main"]
sched = ScheduleQNN_init(fallback_device)
func = sched.visit(func)
mod["main"] = func
seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                relay.transform.RewriteAnnotatedOps(ctx.device_type),
                               ])
with tvm.transform.PassContext(opt_level=3):
    mod = seq(mod)

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize":True}):
    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

