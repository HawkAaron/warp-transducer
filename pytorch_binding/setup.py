from distutils.version import LooseVersion
import os
import platform
import sys
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension


extra_compile_args = ['-fPIC']
if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
    extra_compile_args += ['-std=c++14']
else:
    extra_compile_args += ['-std=c++11']
warp_rnnt_path = "../build"

if torch.cuda.is_available() or "CUDA_HOME" in os.environ:
    enable_gpu = True
else:
    print("Torch was not built with CUDA support, not building GPU extensions.")
    enable_gpu = False

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

if enable_gpu:
    extra_compile_args += ['-DWARPRNNT_ENABLE_GPU']

if "WARP_RNNT_PATH" in os.environ:
    warp_rnnt_path = os.environ["WARP_RNNT_PATH"]
if not os.path.exists(os.path.join(warp_rnnt_path, "libwarprnnt" + lib_ext)):
    print(("Could not find libwarprnnt.so in {}.\n"
           "Build warp-rnnt and set WARP_RNNT_PATH to the location of"
           " libwarprnnt.so (default is '../build')").format(warp_rnnt_path))
    sys.exit(1)
include_dirs = [os.path.realpath('../include')]

setup(
    name='warprnnt_pytorch',
    version="0.1",
    description="PyTorch wrapper for RNN-Transducer",
    url="https://github.com/HawkAaron/warp-transducer",
    author="Mingkun Huang",
    author_email="mingkunhuang95@gmail.com",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='warprnnt_pytorch.warp_rnnt',
            sources=['src/binding.cpp'],
            include_dirs=include_dirs,
            library_dirs=[os.path.realpath(warp_rnnt_path)],
            libraries=['warprnnt'],
            extra_link_args=['-Wl,-rpath,' + os.path.realpath(warp_rnnt_path)],
            extra_compile_args=extra_compile_args), ],
    cmdclass={
        'build_ext': BuildExtension
    })

