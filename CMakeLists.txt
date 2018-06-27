cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

# include included FindTBB module
list(APPEND CMAKE_MODULE_PATH "$ENV{CMAKE_MODULE_PATH}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

project(mapMAP)
include(ExternalProject)

# selection what to build
option(BUILD_MEMSAVE "Use memory-saving mode. Might influence performance negatively." OFF)
option(BUILD_TEST "Build Google Tests for mapMAP modules." ON)
option(BUILD_DEMO "Build a demo, reading SHIMF datasets into mapMAP." ON)

set(TBB_INCLUDE_DIRS "/usr/include" CACHE PATH "Path to where the TBB include files (tbb/) reside")
set(TBB_LIBRARY "/usr/lib" CACHE PATH "Path to where the TBB library files (libtbb.so etc.) reside")

# default to Release build and C++11 mode
set(CMAKE_CXX_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

# set optimization options for GCC on unixoid systems
# uses -march=native to make the compiler set __SSE4_2__, __AVX__ and
# __AVX2__ to fit to the computer
set(CMAKE_CXX_FLAGS "-Wall -march=native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -funroll-loops -mfpmath=sse -flto -g")
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -funroll-loops -mfpmath=sse -flto")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

# augment build parameters for optional builds
if(BUILD_TEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_FOR_TEST")
endif(BUILD_TEST)

if(BUILD_MEMSAVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_MEMORY_SAVE")
endif(BUILD_MEMSAVE)

# if DEMO is built, download a demo dataset (planesweep)
if(BUILD_DEMO)
	file(DOWNLOAD http://download.hrz.tu-darmstadt.de/media/FB20/GCC/project_files/mapmap/planesweep_320_256_96.bin ${PROJECT_SOURCE_DIR}/demo/planesweep_320_256_96.bin)
endif(BUILD_DEMO)

# find location to TBB and set variables
find_package(TBB COMPONENTS tbbmalloc_proxy tbbmalloc tbb REQUIRED)

# include headers
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/mapmap)
include_directories(SYSTEM
    ${TBB_INCLUDE_DIR})

set(header_dir "mapmap/header")
set(impl_dir "mapmap/source")
set(test_dir "test")
set(demo_dir "demo")

# base files
set(header_base
    ${header_dir}/color.h
    ${header_dir}/costs.h
    ${header_dir}/cost_bundle.h
    ${header_dir}/defines.h
    ${header_dir}/graph.h
    ${header_dir}/instance_factory.h
    ${header_dir}/mapmap.h
    ${header_dir}/multilevel.h
    ${header_dir}/parallel_templates.h
    ${header_dir}/termination_criterion.h
    ${header_dir}/timer.h
    ${header_dir}/tree.h
    ${header_dir}/tree_optimizer.h
    ${header_dir}/tree_sampler.h
    ${header_dir}/vector_types.h
    ${header_dir}/vector_math.h)

set(impl_base
    ${impl_dir}/color.impl.h
    ${impl_dir}/costs.impl.h
    ${impl_dir}/cost_bundle.impl.h
    ${impl_dir}/graph.impl.h
    ${impl_dir}/instance_factory.impl.h
    ${impl_dir}/mapmap.impl.h
    ${impl_dir}/multilevel.impl.h
    ${impl_dir}/parallel_templates.impl.h
    ${impl_dir}/timer.impl.h
    ${impl_dir}/tree.impl.h
    ${impl_dir}/tree_optimizer.impl.h
    ${impl_dir}/tree_sampler.impl.h
    ${impl_dir}/vector_math.impl.h)

# cost instance files
set(header_cost_instances
    ${header_dir}/cost_instances/pairwise_potts.h
    ${header_dir}/cost_instances/pairwise_table.h
    ${header_dir}/cost_instances/pairwise_truncated_linear.h
    ${header_dir}/cost_instances/pairwise_truncated_quadratic.h
    ${header_dir}/cost_instances/unary_table.h)

set(impl_cost_instances
    ${impl_dir}/cost_instances/pairwise_potts.impl.h
    ${impl_dir}/cost_instances/pairwise_table.impl.h
    ${impl_dir}/cost_instances/pairwise_truncated_linear.impl.h
    ${impl_dir}/cost_instances/pairwise_truncated_quadratic.impl.h
    ${impl_dir}/cost_instances/unary_table.impl.h)

# optimizer
set(header_optimizer_instances
    ${header_dir}/optimizer_instances/envelope_instances/pairwise_potts_envelope.h
    ${header_dir}/optimizer_instances/envelope_instances/pairwise_truncated_linear_envelope.h
    ${header_dir}/optimizer_instances/envelope_instances/pairwise_truncated_quadratic_envelope.h
    ${header_dir}/optimizer_instances/dp_node.h
    ${header_dir}/optimizer_instances/dp_node_solver.h
    ${header_dir}/optimizer_instances/dp_node_solver_factory.h
    ${header_dir}/optimizer_instances/dynamic_programming.h
    ${header_dir}/optimizer_instances/envelope.h)

set(impl_optimizer_instances
    ${impl_dir}/optimizer_instances/envelope_instances/pairwise_potts_envelope.impl.h
    ${impl_dir}/optimizer_instances/envelope_instances/pairwise_truncated_linear_envelope.impl.h
    ${impl_dir}/optimizer_instances/envelope_instances/pairwise_truncated_quadratic_envelope.impl.h
    ${impl_dir}/optimizer_instances/dp_node_solver.impl.h
    ${impl_dir}/optimizer_instances/dp_node_solver_factory.impl.h
    ${impl_dir}/optimizer_instances/dynamic_programming.impl.h)

# multilevel instance files
set(header_multilevel_instances
    ${header_dir}/multilevel_instances/group_same_label.h)

set(impl_multilevel_instances
    ${impl_dir}/multilevel_instances/group_same_label.impl.h)

# termination criterion instances
set(header_termination_instances
    ${header_dir}/termination_instances/stop_after_iterations.h
    ${header_dir}/termination_instances/stop_after_time.h
    ${header_dir}/termination_instances/stop_when_flat.h
    ${header_dir}/termination_instances/stop_when_returns_diminish.h)

set(impl_termination_instances
    ${impl_dir}/termination_instances/stop_after_iterations.impl.h
    ${impl_dir}/termination_instances/stop_after_time.impl.h
    ${impl_dir}/termination_instances/stop_when_flat.impl.h
    ${impl_dir}/termination_instances/stop_when_returns_diminish.impl.h)

# tree sampler instances
set(header_tree_sampler_instances
    ${header_dir}/tree_sampler_instances/lock_free_tree_sampler.h
    ${header_dir}/tree_sampler_instances/optimistic_tree_sampler.h)

set(impl_tree_sampler_instances
    ${impl_dir}/tree_sampler_instances/lock_free_tree_sampler.impl.h
    ${impl_dir}/tree_sampler_instances/optimistic_tree_sampler.impl.h)

# test files
set(header_test
    ${test_dir}/util_test.h)

set(impl_test
    ${test_dir}/util_test.impl.h)

set(src_test
    ${test_dir}/test_coloring.cc
    ${test_dir}/test_coordinate_set.cc
    ${test_dir}/test_dynamic_programming.cc
    ${test_dir}/test_envelope.cc
    ${test_dir}/test_graph.cc
    ${test_dir}/test_multilevel.cc
    ${test_dir}/test_spanning_tree.cc
    ${test_dir}/test_vector_math.cc)

set_source_files_properties(${header_base} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${header_cost_instances} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${header_optimizer_instances} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${header_multilevel_instances} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${header_termination_instances} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${tree_sampler_instances} PROPERTIES HEADER_FILE_ONLY TRUE)

source_group("Base" FILES
    ${header_base}
    ${impl_base})
source_group("Costs" FILES
    ${header_cost_instances}
    ${impl_cost_instances})
source_group("Optimizer" FILES
    ${header_optimizer_instances}
    ${impl_optimizer_instances})
source_group("Multilevel" FILES
    ${header_multilevel_instances}
    ${impl_multilevel_instances})
source_group("Termination" FILES
    ${header_termination_instances}
    ${impl_termination_instances})
source_group("TreeSampler" FILES
    ${header_tree_sampler_instances}
    ${impl_tree_sampler_instances})
source_group("Coloring" FILES
    ${header_coloring_instances}
    ${impl_coloring_instances})
source_group("Test" FILES
    ${header_test}
    ${impl_test}
    ${src_test})

# CMake as dependency: download and build
externalproject_add(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG master
    CMAKE_ARGS -DBUILD_GTEST=ON -DBUILD_SHARED_LIBS=ON
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_TEST ON)

ExternalProject_Get_Property(googletest source_dir)
include_directories(SYSTEM
    ${source_dir}/googletest/include)
ExternalProject_Get_Property(googletest binary_dir)
link_directories(
    ${binary_dir}/googlemock/gtest)

# build tests
if(BUILD_TEST)

add_executable(mapmap_test
    ${src_test})
add_dependencies(mapmap_test googletest)
target_link_libraries(mapmap_test
    ${TBB_LIBRARIES}
    gtest
    gtest_main
    pthread)
endif(BUILD_TEST)

# build demo executable solver
if(BUILD_DEMO)
add_executable(mapmap_demo
    ${src_base}
    ${demo_dir}/mapmap_demo.cc)

if(BUILD_TEST)
    add_dependencies(mapmap_demo googletest)
endif(BUILD_TEST)

target_link_libraries(mapmap_demo
    ${TBB_LIBRARIES})
endif(BUILD_DEMO)
