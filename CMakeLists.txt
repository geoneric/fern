cmake_minimum_required(VERSION 2.8)

project(Fern)

set(${PROJECT_NAME}_ORGANISATION_NAME "Geoneric")
set(${PROJECT_NAME}_COPYRIGHT "2010-2015 Geoneric")
set(${PROJECT_NAME}_PROJECT_NAME "${PROJECT_NAME}")
set(${PROJECT_NAME}_HOMEPAGE "http://www.geoneric.eu")
set(${PROJECT_NAME}_MAJOR_VERSION "0")
set(${PROJECT_NAME}_MINOR_VERSION "0")
set(${PROJECT_NAME}_PATCH_VERSION "0")
set(${PROJECT_NAME}_PRE_RELEASE_VERSION "-rc.1")
set(${PROJECT_NAME}_VERSION
    "${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION}.${${PROJECT_NAME}_PATCH_VERSION}${${PROJECT_NAME}_PRE_RELEASE_VERSION}")

set(CMAKE_MODULE_PATH
    $ENV{CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/environment/cmake
)
include(Fern)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_run_prolog.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/test_run_prolog.sh
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_run_epilog.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/test_run_epilog.sh
)

# Order matters. some doc targets depend on source targets.
add_subdirectory(environment)
add_subdirectory(source)
add_subdirectory(document)

install(
    EXPORT fern_targets
    DESTINATION lib/fern
)
