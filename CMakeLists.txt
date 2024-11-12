cmake_minimum_required(VERSION 3.10)

project(AMGMatchGPU C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Wall -fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O0")

set(SRC_DIR src)
set(INCLUDE_DIR include)

include_directories(${INCLUDE_DIR})

file(GLOB SRCS "${SRC_DIR}/*.c")

add_executable(test ${SRCS})

set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build
)
