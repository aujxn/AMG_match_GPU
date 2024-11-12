# AMG Match GPU

A reimplementation of my AMG match research library in `C` utilizing no dependencies (except CUDA) with goals of running on GPUs one day.

## Build and Run Tests

To build and run tests (will require some test matrices... see: `src/test.c`):
```
mkdir -p build
cd build
cmake ..
make
./build/test
```

