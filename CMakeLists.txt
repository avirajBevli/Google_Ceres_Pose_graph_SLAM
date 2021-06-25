cmake_minimum_required(VERSION 2.8)

project(helloworld)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(data_gen1 data_gen1.cpp)
target_link_libraries(data_gen1 ${CERES_LIBRARIES})

add_executable(slam slam.cpp)
target_link_libraries(slam ${CERES_LIBRARIES})