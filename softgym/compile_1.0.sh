cd /workspace/RL-VLM-F/softgym/PyFlex/bindings
rm -rf build
mkdir build && cd build

export CUDA_BIN_PATH=/usr/local/cuda-11.3
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_BIN_PATH

cmake -DPYBIND11_PYTHON_VERSION=3.9 ..
make -j$(nproc)

# Set Python paths so Python can see pyflex and softgym
export PYFLEXROOT=/workspace/RL-VLM-F/softgym/PyFlex
export PYTHONPATH=/workspace/RL-VLM-F:/worksapce/RL-VLM-F/softgym:$PYFLEXROOT/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
