cmake_minimum_required(VERSION 3.1)

project(autojac)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE Debug)

## Find catkin macros and libraries
find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# add cmake functions
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include (add_doxygen_compile)
include (add_gtest_compile)

# add compile options
option(BUILD_TEST "Build tests" OFF)
option(BUILD_DOC "Build documentation" OFF)
option(BUILD_ALL "Build all" OFF)

if(BUILD_ALL)
  set(BUILD_TEST ON)
  set(BUILD_DOC ON)
endif()

# setup catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  DEPENDS EIGEN3
)

###########
## Build ##
###########


## Specify additional locations of header files
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
# see https://answers.ros.org/question/223866/cmakecatkin-how-to-export-imported-target/
add_library(cppad_lib SHARED IMPORTED)
set_property(TARGET cppad_lib PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/lib/libcppad_lib.so)

# define targets
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${PROJECT_NAME} cppad_lib ${catkin_LIBRARIES} ${CMAKE_DL_LIBS})



#############
## Install ##
#############

## Copy CppAD Lib
file(COPY ${PROJECT_SOURCE_DIR}/lib/
  DESTINATION ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_LIB_DESTINATION}
  FILES_MATCHING PATTERN "libcppad_lib*"
)

install(FILES ${CPPAD_PATH} DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

## Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(DIRECTORY include/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

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