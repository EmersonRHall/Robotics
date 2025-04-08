cmake_minimum_required(VERSION 3.10)
project(FeatureTracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_CFLAGS_OTHER})

add_executable(FeatureTracking main.cpp)
target_link_libraries(FeatureTracking ${OPENCV_LIBRARIES})
