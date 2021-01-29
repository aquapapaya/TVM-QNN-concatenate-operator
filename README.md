# TVM-QNN-concatenate-operator
## CallNode is not considered when canonicalizing qnn.concatenate
* The TVM commit [da27e6d](https://github.com/apache/tvm/tree/da27e6d9a466263a9a0025aba92086a8bf837edb) is tested.
* In [test_yolo.py](test_yolo.py), we dispatch qnn.concatenate to generate CallNode: run <code>python3 test_yolo.py</code> to reproduce the [problem](tuple_input_scales-failed.txt). 
* A fixed [qnn.concatenate](concatenate.cc) is provided. Please copy it to src/relay/qnn/op of TVM, re-build TVM, and then you can get input scale and zero point of CallNode.
