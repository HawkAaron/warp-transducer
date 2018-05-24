# TensorFlow binding for WarpRNNT

This package provides TensorFlow kernels that wrap the WarpRNNT
library.

## Installation

To build the kernels it is necessary to have the TensorFlow source
code available, since TensorFlow doesn't currently install the
necessary headers to handle the SparseTensor that the CTCLoss op uses
to input the labels.  You can retrieve the TensorFlow source from
github.com:

```bash
git clone https://github.com/tensorflow/tensorflow.git
```

Tell the build scripts where you have the TensorFlow source tree by
setting the `TENSORFLOW_SRC_PATH` environment variable:

```bash
export TENSORFLOW_SRC_PATH=/path/to/tensorflow
```

`WARP_RNNT_PATH` should be set to the location of a built WarpRNNT
(i.e. `libwarprnnt.so`).  This defaults to `../build`, so from within a
new warp-rnnt clone you could build WarpRNNT like this:

```bash
mkdir build; cd build
cmake ..
make
```

Otherwise, set `WARP_RNNT_PATH` to wherever you have `libwarprnnt.so`
installed. If you have a GPU, you should also make sure that
`CUDA_HOME` is set to the home cuda directory (i.e. where
`include/cuda.h` and `lib/libcudart.so` live).

You should now be able to use `setup.py` to install the package into
your current Python environment:

```bash
python setup.py install
```

You can run a few unit tests with `setup.py` as well if you want:

```bash
python setup.py test
```

## Using the kernels

First import the module:

```python
import warprnnt_tensorflow
```

The WarpRNNT op is available via the `warprnnt_tensorflow.rnnt` function:

```python
costs = warprnnt_tensorflow.rnnt(trans_acts, pred_acts, flat_labels, label_lengths, input_lengths)
```

The `trans_acts` and `pred_acts` inputs are 3 dimensional Tensors and all the others
are single dimension Tensors.  See the main WarpRNNT documentation for
more information.