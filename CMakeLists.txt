cmake_minimum_required(VERSION 3.10)
project(active_set_method)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EnableC++XX)
include(EnableSanitizers)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(least_squares_active_set INTERFACE)
target_include_directories(least_squares_active_set INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(least_squares_active_set INTERFACE Eigen3::Eigen)
target_compile_features(least_squares_active_set INTERFACE cxx_std_17)
target_compile_options(least_squares_active_set INTERFACE
    -Wall -Wextra -Wpedantic -fdiagnostics-show-option
)

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(main PRIVATE least_squares_active_set Eigen3::Eigen)
target_compile_features(main PRIVATE cxx_std_17)
target_compile_options(main PRIVATE
    -Wall -Wextra -Wpedantic -fdiagnostics-show-option
)
