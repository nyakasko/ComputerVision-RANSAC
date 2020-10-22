# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("sensorfusion_assign1")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( sensorfusion_assign1 sensorfusion_assign1/sensorfusion_assign1.cpp )
target_link_libraries( sensorfusion_assign1 ${OpenCV_LIBS} )

# Include sub-projects.
add_subdirectory ("sensorfusion_assign1")
