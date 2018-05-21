# warp-transducer
A fast parallel implementation of RNN Transducer, on both CPU and GPU.

## GPU Performance
Benchmarked on a GeForce GTX 1080 Ti GPU.

| **T=150, L=40, A=28** | **warp-transducer** |
| --------------------- | ------------------- |
|         N=1           |      8.98 ms        |
|         N=16          |      12.01 ms       |
|         N=32          |      14.96 ms       |
|         N=64          |      16.49 ms       |
|         N=128         |      20.84 ms       |

| **T=150, L=20, A=5000** | **warp-transducer** |
| ----------------------- | ------------------- |
|         N=1             |      6.51 ms        |
|         N=16            |      28.76 ms       |
|         N=32            |      44.22 ms       |
|         N=64            |      81.08 ms       |
|         N=128           |      165.41 ms      |

| **T=1500, L=300, A=50** | **warp-transducer** |
| ----------------------- | ------------------- |
|         N=1             |      508.84 ms      |
|         N=16            |      838.36 ms      |
|         N=32            |      1075.39 ms     |
|         N=64            |      1579.13 ms     |
|         N=128           |      2298.22 ms     |

| **T=320, L=80, A=30000** | **warp-transducer** |
| ------------------------ | ------------------- |
|         N=1              |      96.36 ms       |
|         N=16             |      1049.82 ms     |
|         N=32             |      2061.54 ms     |
|         N=64             |      4082.06 ms     |
|         N=128            |      -              |

## Installation

Install [PyTorch](https://github.com/pytorch/pytorch#installation).

`WARP_RNNT_PATH` should be set to the location of a built WarpRNNT
(i.e. `libwarprnnt.so`).  This defaults to `../build`, so from within a
new warp-rnnt clone you could build WarpRNNT like this:

```bash
git clone https://github.com/HawkAaron/warp-transducer
cd warp-transducer
mkdir build; cd build
cmake ..
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
* [Baidu warp-ctc](https://github.com/baidu-research/warp-ctc)
* [Awni implementation of transducer](https://github.com/awni/transducer)

