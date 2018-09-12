"""setup.py script for warp-rnnt TensorFlow wrapper"""

from __future__ import print_function

import os
import platform
import re
import setuptools
import sys
import unittest
from setuptools.command.build_ext import build_ext as orig_build_ext
from distutils.version import LooseVersion

# We need to import tensorflow to find where its include directory is.
try:
    import tensorflow as tf
except ImportError:
    raise RuntimeError("Tensorflow must be installed to build the tensorflow wrapper.")

if "CUDA_HOME" not in os.environ:
    print("CUDA_HOME not found in the environment so building "
          "without GPU support. To build with GPU support "
          "please define the CUDA_HOME environment variable. "
          "This should be a path which contains include/cuda.h",
          file=sys.stderr)
    enable_gpu = False
else:
    enable_gpu = True

'''
if "TENSORFLOW_SRC_PATH" not in os.environ:
    print("Please define the TENSORFLOW_SRC_PATH environment variable.\n"
          "This should be a path to the Tensorflow source directory.",
          file=sys.stderr)
    sys.exit(1)
'''

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

warp_rnnt_path = "../build"
if "WARP_RNNT_PATH" in os.environ:
    warp_rnnt_path = os.environ["WARP_RNNT_PATH"]
if not os.path.exists(os.path.join(warp_rnnt_path, "libwarprnnt"+lib_ext)):
    print(("Could not find libwarprnnt.so in {}.\n"
           "Build warp-rnnt and set WARP_RNNT_PATH to the location of"
           " libwarprnnt.so (default is '../build')").format(warp_rnnt_path),
          file=sys.stderr)
    sys.exit(1)

root_path = os.path.realpath(os.path.dirname(__file__))

tf_include = tf.sysconfig.get_include()
tf_src_dir = tf.sysconfig.get_lib() # os.environ["TENSORFLOW_SRC_PATH"]
tf_includes = [tf_include, tf_src_dir]
warp_rnnt_includes = [os.path.join(root_path, '../include')]
include_dirs = tf_includes + warp_rnnt_includes

if LooseVersion(tf.__version__) >= LooseVersion('1.4'):
    include_dirs += [os.path.join(tf_include, 'external/nsync/public')]

extra_compile_args = ['-std=c++11', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=0']
# current tensorflow code triggers return type errors, silence those for now
extra_compile_args += ['-Wno-return-type']

extra_link_args = []
if LooseVersion(tf.__version__) >= LooseVersion('1.4'):
    if os.path.exists(os.path.join(tf_src_dir, 'libtensorflow_framework.so')):
        extra_link_args = ['-L' + tf_src_dir, '-ltensorflow_framework']

if (enable_gpu):
    extra_compile_args += ['-DWARPRNNT_ENABLE_GPU']
    include_dirs += [os.path.join(os.environ["CUDA_HOME"], 'include')]

    # mimic tensorflow cuda include setup so that their include command work
    if not os.path.exists(os.path.join(root_path, "include")):
        os.mkdir(os.path.join(root_path, "include"))

    cuda_inc_path = os.path.join(root_path, "include/cuda")
    if not os.path.exists(cuda_inc_path) or os.readlink(cuda_inc_path) != os.environ["CUDA_HOME"]:
        if os.path.exists(cuda_inc_path):
            os.remove(cuda_inc_path)
        os.symlink(os.environ["CUDA_HOME"], cuda_inc_path)
    include_dirs += [os.path.join(root_path, 'include')]

# Ensure that all expected files and directories exist.
for loc in include_dirs:
    if not os.path.exists(loc):
        print(("Could not find file or directory {}.\n"
               "Check your environment variables and paths?").format(loc),
              file=sys.stderr)
        sys.exit(1)

lib_srcs = ['src/warprnnt_op.cc']

ext = setuptools.Extension('warprnnt_tensorflow.kernels',
                           sources = lib_srcs,
                           language = 'c++',
                           include_dirs = include_dirs,
                           library_dirs = [warp_rnnt_path],
                           runtime_library_dirs = [os.path.realpath(warp_rnnt_path)],
                           libraries = ['warprnnt'],
                           extra_compile_args = extra_compile_args,
                           extra_link_args = extra_link_args)

class build_tf_ext(orig_build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        orig_build_ext.build_extensions(self)

def discover_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

# Read the README.md file for the long description. This lets us avoid
# duplicating the package description in multiple places in the source.
README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
with open(README_PATH, "r") as handle:
    # Extract everything between the first set of ## headlines
    LONG_DESCRIPTION = re.search("#.*([^#]*)##", handle.read()).group(1).strip()

setuptools.setup(
    name = "warprnnt_tensorflow",
    version = "0.1",
    description = "TensorFlow wrapper for warp-transducer",
    url="https://github.com/HawkAaron/warp-transducer",
    long_description = LONG_DESCRIPTION,
    author = "Mingkun Huang",
    author_email = "mingkunhuang95@gmail.com",
    license = "Apache",
    packages = ["warprnnt_tensorflow"],
    ext_modules = [ext],
    cmdclass = {'build_ext': build_tf_ext},
    test_suite = 'setup.discover_test_suite',
)
