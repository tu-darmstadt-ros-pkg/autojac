cmake_minimum_required(VERSION 3.3)

project(autojac)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE Debug)

## Find ament macros and libraries
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

# add cmake functions
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include (add_doxygen_compile)
include (add_gtest_compile)

# add compile options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

option(BUILD_TEST "Build tests" OFF)
option(BUILD_DOC "Build documentation" OFF)
option(BUILD_ALL "Build all" OFF)

if(BUILD_ALL)
  set(BUILD_TEST ON)
  set(BUILD_DOC ON)
endif()

###########
## Build ##
###########

## Specify source and header files
set(HEADERS
    include/${PROJECT_NAME}/dense_jacobian_symbolic.h
    include/${PROJECT_NAME}/dense_jacobian_code_gen.h
    include/${PROJECT_NAME}/sparse_jacobian_symbolic.h
    include/${PROJECT_NAME}/sparse_jacobian_code_gen.h
    include/${PROJECT_NAME}/sparsity_pattern_handler.h
    include/${PROJECT_NAME}/function_generation_util.h
    include/${PROJECT_NAME}/types.h
    include/${PROJECT_NAME}/cppad_error_handling.h
)

set(SOURCES
    src/dense_jacobian_symbolic.cpp
    src/dense_jacobian_code_gen.cpp
    src/sparse_jacobian_symbolic.cpp
    src/sparse_jacobian_code_gen.cpp
    src/sparsity_pattern_handler.cpp
    src/function_generation_util.cpp
    src/cppad_error_handling.cpp
)

# setup cppad library
add_library(cppad SHARED IMPORTED)
set_property(TARGET cppad PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/lib/libcppad_lib.so)


# define targets
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
set_target_properties(${PROJECT_NAME} PROPERTIES
LINKER_LANGUAGE CXX
)
target_link_libraries(${PROJECT_NAME} PRIVATE cppad ${CMAKE_DL_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
    ${EIGEN3_INCLUDE_DIR}
  )

#############
## Install ##
#############
install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY
   ${PROJECT_SOURCE_DIR}/lib/
  DESTINATION lib
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_libraries(${PROJECT_NAME})
ament_export_include_directories(include)
ament_export_dependencies(Eigen3)

#############
## Testing ##
#############

set(TEST_SOURCES
    test/src/dense_jacobian_symbolic_test.cpp
    test/src/dense_jacobian_code_gen_test.cpp
    test/src/sparse_jacobian_symbolic_test.cpp
    test/src/sparse_jacobian_code_gen_test.cpp
)

add_gtest_compile()

##########
## DOCS ##
##########

add_doxygen_compile()

ament_package()