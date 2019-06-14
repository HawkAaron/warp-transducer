# warp-transducer
A fast parallel implementation of RNN Transducer (Graves 2013 joint network), on both CPU and GPU.

[GPU implementation is now available for Graves2012 add network.](https://github.com/HawkAaron/warp-transducer/tree/add_network)

## GPU Performance
Benchmarked on a GeForce GTX 1080 Ti GPU.

| **T=150, L=40, A=28** | **warp-transducer** |
| --------------------- | ------------------- |
|         N=1           |      8.51 ms        |
|         N=16          |      11.43 ms       |
|         N=32          |      12.65 ms       |
|         N=64          |      14.75 ms       |
|         N=128         |      19.48 ms       |

| **T=150, L=20, A=5000** | **warp-transducer** |
| ----------------------- | ------------------- |
|         N=1             |      4.79 ms        |
|         N=16            |      24.44 ms       |
|         N=32            |      41.38 ms       |
|         N=64            |      80.44 ms       |
|         N=128           |      51.46 ms       |

<!-- | **T=1500, L=300, A=50** | **warp-transducer** |
| ----------------------- | ------------------- |
|         N=1             |      570.33 ms      |
|         N=16            |      768.57 ms      |
|         N=32            |      955.05 ms      |
|         N=64            |      569.34 ms      |
|         N=128           |      -              |
 -->

## Interface
The interface is in `include/rnnt.h`. It supports CPU or GPU execution, and you can specify OpenMP parallelism
if running on the CPU, or the CUDA stream if running on the GPU. We took care to ensure that the library does not 
preform memory allocation internally, in oder to avoid synchronizations and overheads caused by memory allocation.
**Please be carefull if you use the RNNTLoss CPU version, log_softmax should be manually called before the loss function.
(For pytorch binding, this is optionally handled by tensor device.)**

## Compilation
warp-transducer has been tested on Ubuntu 16.04 and CentOS 7. Windows is not supported at this time.

First get the code:
```bash
git clone https://github.com/HawkAaron/warp-transducer
cd warp-transducer
```
create a build directory:
```bash
mkdir build
cd build
```
if you have a non standard CUDA install, add `-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda` option to `cmake` so that CMake detects CUDA.

Run cmake and build:
```bash
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME ..
make
```
if it logs
```
-- cuda found TRUE
-- Building shared library with no GPU support
```
please run `rm CMakeCache.txt` and cmake again.

The C library should now be built along with test executables. If CUDA was detected, then `test_gpu` will be built;
`test_cpu` will always be built.

## Test
To run the tests, make sure the CUDA libraries are in `LD_LIBRARY_PATH` (DYLD_LIBRARY_PATH for OSX).

## Contributing
We welcome improvements from the community, please feel free to submit pull requests.

## Reference
* [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* [SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1303.5778.pdf)
* [Baidu warp-ctc](https://github.com/baidu-research/warp-ctc)
* [Awni implementation of transducer](https://github.com/awni/transducer)

