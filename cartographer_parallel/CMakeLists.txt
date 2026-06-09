cmake_minimum_required(VERSION 3.10)
project(cartographer_parallel LANGUAGES CXX)

option(BUILD_CUDA_TASK "Build CUDA score_all and CUDA matcher node" ON)
option(BUILD_GPU_TASK "Alias for BUILD_CUDA_TASK" ${BUILD_CUDA_TASK})

if(NOT BUILD_CUDA_TASK OR NOT BUILD_GPU_TASK)
  set(BUILD_CUDA_TASK OFF)
  set(BUILD_GPU_TASK OFF)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)

set(CARTOGRAPHER_PARALLEL_PACKAGE_LIBS assignment_cpu_lib)
if(BUILD_CUDA_TASK)
  list(APPEND CARTOGRAPHER_PARALLEL_PACKAGE_LIBS
    assignment_cuda_lib
    cuda_fast_matcher_lib
  )
else()
  list(APPEND CARTOGRAPHER_PARALLEL_PACKAGE_LIBS
    fast_matcher_lib
    cpu_assignment_lib
    cpu_fast_matcher_lib
  )
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${CARTOGRAPHER_PARALLEL_PACKAGE_LIBS}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(assignment_cpu_lib
  src/score_all.cpp
)
target_link_libraries(assignment_cpu_lib
  ${catkin_LIBRARIES}
)

set(CARTOGRAPHER_PARALLEL_INSTALL_TARGETS
  assignment_cpu_lib
)

if(BUILD_CUDA_TASK)
  enable_language(CUDA)

  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(CMAKE_CUDA_ARCHITECTURES 53)
  else()
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -gencode arch=compute_53,code=sm_53")
  endif()

  add_library(assignment_cuda_lib
    src/cuda_score_all.cu
    src/cuda_score_all_ver1.cu
    src/cuda_score_all_ver2.cu
    src/cuda_score_all_ver3.cu
    src/cuda_score_all_ver4.cu
    src/cuda_score_all_ver5.cu
    src/cuda_score_all_ver6.cu
    src/cuda_score_all_ver7.cu
    src/cuda_score_all_ver8.cu
  )
  target_include_directories(assignment_cuda_lib PUBLIC
    include
    ${catkin_INCLUDE_DIRS}
  )
  target_link_libraries(assignment_cuda_lib
    ${catkin_LIBRARIES}
  )
  set_target_properties(assignment_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set_target_properties(assignment_cuda_lib PROPERTIES
      CUDA_ARCHITECTURES 53
    )
  endif()

  add_library(cuda_fast_matcher_lib
    src/cuda_fast_matcher.cpp
  )
  target_include_directories(cuda_fast_matcher_lib PUBLIC
    include
    ${catkin_INCLUDE_DIRS}
  )
  target_link_libraries(cuda_fast_matcher_lib
    assignment_cpu_lib
    assignment_cuda_lib
    ${catkin_LIBRARIES}
  )
  target_compile_definitions(cuda_fast_matcher_lib PRIVATE
    CARTOGRAPHER_PARALLEL_USE_CUDA
  )

  add_executable(fast_correlative_node_print
    src/fast_correlative_node_print.cpp
  )
  target_link_libraries(fast_correlative_node_print
    cuda_fast_matcher_lib
    ${catkin_LIBRARIES}
  )

  add_executable(fast_correlative_node
    src/fast_correlative_node_print.cpp
  )
  target_link_libraries(fast_correlative_node
    cuda_fast_matcher_lib
    ${catkin_LIBRARIES}
  )

  list(APPEND CARTOGRAPHER_PARALLEL_INSTALL_TARGETS
    assignment_cuda_lib
    cuda_fast_matcher_lib
    fast_correlative_node
    fast_correlative_node_print
  )
else()
  find_package(OpenMP)

  add_library(cpu_assignment_lib
    src/cpu_score_all.cpp
  )
  target_include_directories(cpu_assignment_lib PUBLIC
    include
    ${catkin_INCLUDE_DIRS}
  )
  target_link_libraries(cpu_assignment_lib
    ${catkin_LIBRARIES}
  )
  if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_assignment_lib
      OpenMP::OpenMP_CXX
    )
  endif()

  add_library(fast_matcher_lib
    src/fast_matcher_print.cpp
  )
  target_link_libraries(fast_matcher_lib
    assignment_cpu_lib
    ${catkin_LIBRARIES}
  )

  add_library(cpu_fast_matcher_lib
    src/cpu_fast_matcher.cpp
  )
  target_include_directories(cpu_fast_matcher_lib PUBLIC
    include
    ${catkin_INCLUDE_DIRS}
  )
  target_link_libraries(cpu_fast_matcher_lib
    cpu_assignment_lib
    assignment_cpu_lib
    ${catkin_LIBRARIES}
  )
  if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_fast_matcher_lib
      OpenMP::OpenMP_CXX
    )
  endif()

  add_executable(fast_correlative_node_print
    src/fast_correlative_node_print.cpp
  )
  target_link_libraries(fast_correlative_node_print
    fast_matcher_lib
    ${catkin_LIBRARIES}
  )

  add_executable(fast_correlative_node
    src/fast_correlative_node_print.cpp
  )
  target_link_libraries(fast_correlative_node
    fast_matcher_lib
    ${catkin_LIBRARIES}
  )

  add_executable(cpu_fast_correlative_node
    src/cpu_fast_correlative_node.cpp
  )
  target_link_libraries(cpu_fast_correlative_node
    cpu_fast_matcher_lib
    ${catkin_LIBRARIES}
  )

  list(APPEND CARTOGRAPHER_PARALLEL_INSTALL_TARGETS
    cpu_assignment_lib
    fast_matcher_lib
    cpu_fast_matcher_lib
    fast_correlative_node_print
    fast_correlative_node
    cpu_fast_correlative_node
  )
endif()

install(TARGETS
  ${CARTOGRAPHER_PARALLEL_INSTALL_TARGETS}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch maps bags
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
