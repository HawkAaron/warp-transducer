# warp-transducer
A fast parallel implementation of RNN Transducer (Graves 2013 joint network), on both CPU and GPU.

[GPU implementation is now available for Graves2012 add network.](https://github.com/HawkAaron/warp-transducer/tree/add_network)

## Installation

Install [PyTorch](https://github.com/pytorch/pytorch#installation).

`WARP_RNNT_PATH` should be set to the location of a built WarpRNNT
(i.e. `libwarprnnt.so`).  This defaults to `../build`, so from within a
new warp-rnnt clone you could build WarpRNNT like this:

```bash
git clone https://github.com/HawkAaron/warp-transducer
cd warp-transducer
mkdir build; cd build
cmake -D WITH_GPU=OFF ..
make
```

Otherwise, set `WARP_RNNT_PATH` to wherever you have `libwarprnnt.so`
installed. If you have a GPU, you should also make sure that
`CUDA_HOME` is set to the home cuda directory (i.e. where
`include/cuda.h` and `lib/libcudart.so` live). For example:

```
export CUDA_HOME="/usr/local/cuda"
```

Now install the bindings:
```
cd pytorch_binding
python setup.py install
```

If you try the above and get a dlopen error on OSX with anaconda3 (as recommended by pytorch):
```
cd ../pytorch_binding
python setup.py install
cd ../build
cp libwarprnnt.dylib /Users/$WHOAMI/anaconda3/lib
```
This will resolve the library not loaded error. This can be easily modified to work with other python installs if needed.

## Reference
* [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* [SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1303.5778.pdf)
* [Baidu warp-ctc](https://github.com/baidu-research/warp-ctc)
* [Awni implementation of transducer](https://github.com/awni/transducer)

