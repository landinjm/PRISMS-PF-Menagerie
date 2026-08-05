##
#  CMake script for PRISMS-PF-Menagerie
##

cmake_minimum_required(VERSION 3.25)
cmake_policy(VERSION 3.25)

project(PRISMS-PF-Menagerie CXX)

# Set the build type to debug is not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# List of all the application
set(
  applications
  snow_crystal
  surface_diffusion/explicit
)

# Create a target for each example and feed them into a big target
set(application_targets)
foreach(application IN LISTS applications)
  string(REPLACE "/" "_" application_name ${application})

  add_custom_target(
    ${application_name}
    COMMAND
      ${CMAKE_COMMAND} --log-level=NOTICE -S ${CMAKE_SOURCE_DIR}/${application} -B
      ${CMAKE_BINARY_DIR}/${application} -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}
    COMMAND
      ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/${application}
  )

  # TODO: Bother with an install target? To be honest this bit is on here as
  # a CI thing.

  list(APPEND application_targets ${application_name})
endforeach()
add_custom_target(build_applications ALL DEPENDS ${application_targets})
