cmake_minimum_required(VERSION 3.22)
project(standalone)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pyarrow as pa; print(pa.get_include())"
        OUTPUT_VARIABLE PATH_OUTPUT
        ERROR_VARIABLE PATH_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pyarrow as pa; print(pa.get_library_dirs()[0])"
        OUTPUT_VARIABLE LIB_PATH_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Store the path in a CMake variable
set(PYARROW_LIB_PATH "${LIB_PATH_OUTPUT}" CACHE INTERNAL "Path to pyarrow library directory")

# Store the path in a CMake variable
set(PYARROW_INCLUDE_PATH "${PATH_OUTPUT}" CACHE INTERNAL "Path to pyarrow include directory")

execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE Pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Pybind11_DIR: ${Pybind11_DIR}")
message(STATUS "PYARROW_LIB_PATH: ${PYARROW_LIB_PATH}")

find_package(pybind11 REQUIRED PATHS ${Pybind11_DIR})

add_executable(standalone standalone.cpp)
target_include_directories(standalone PRIVATE ${PYARROW_INCLUDE_PATH})
target_link_libraries(standalone PRIVATE pybind11::embed arrow_python arrow)
target_link_directories(standalone PRIVATE ${PYARROW_LIB_PATH})
