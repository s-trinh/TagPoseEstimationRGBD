project(TagPoseEstimationRGBD)

cmake_minimum_required(VERSION 3.0)

find_package(VISP REQUIRED visp_core visp_detection visp_io visp_gui visp_sensor)
include_directories(${VISP_INCLUDE_DIRS})

# set the list of source files
set(src_cpp
  src/tag_pose_rgbd.cpp
)

foreach(cpp ${src_cpp})
  get_filename_component(target ${cpp} NAME_WE)
  add_executable(${target} ${cpp}
  )
  target_link_libraries(${target} ${VISP_LIBRARIES})
endforeach()
