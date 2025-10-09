cd PyFlex/bindings
rm -rf build
mkdir build
cd build

export CUDA_BIN_PATH=/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_BIN_PATH

cmake -DPYBIND11_PYTHON_VERSION=3.9 ..
make -j$(nproc)
