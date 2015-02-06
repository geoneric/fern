include(CheckCXXSourceRuns)


macro(add_parser_generation_command
        BASENAME)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.map.in
        ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.map
    )

    add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}-pskel.hxx
            ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}-pskel.cxx
        COMMAND
            ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
                --type-map ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.map
                ${ARGN}
                ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.xsd
                DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.map
            ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.xsd
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endmacro()


# Targets can be added conditionally. When Fern components are selected during
# configuration, certain variables are set that tell us whether or not to
# include a target or group of targets. See FernConfiguration.cmake.
# TARGET_NAME   : Name of target, snake_case.
# DIRECTORY_NAME: Name of subdirectory containing the target.
macro(add_target_conditionally
        TARGET_NAME
        DIRECTORY_NAME)
    # Determine name of variable to check.
    string(TOUPPER ${TARGET_NAME} VARIABLE_NAME)
    set(VARIABLE_NAME "FERN_FERN_${VARIABLE_NAME}_REQUIRED")

    # Make sure it is defined.
    if(NOT DEFINED ${VARIABLE_NAME})
        message(SEND_ERROR "Variable ${VARIABLE_NAME} is not defined")
    endif()

    # Evaluate the variable and include target if result is true.
    if(${VARIABLE_NAME})
        add_subdirectory(${DIRECTORY_NAME})
    endif()
endmacro()


# Create a static library and an object library.
# BASENAME: Name of static library to create. The object library will be 
#           named ${BASENAME}_objects.
# SOURCES : Sources that are part of the libraries. Any argument that comes
#           after the BASENAME is treated as a source.
macro(add_library_and_object_library
        BASENAME)
    set(SOURCES ${ARGN})
    add_library(${BASENAME}
        ${SOURCES}
    )
    add_library(${BASENAME}_objects OBJECT
        ${SOURCES}
    )
endmacro()


# Verify that headers are self-sufficient: they include the headers they need.
# OFFSET_PATHNAME : Pathname to root of headers.
# INCLUDES        : Pathnames where included headers can be found.
# LIBRARIES       : Libraries to link.
# FLAGS           : Compiler flags.
# HEADER_PATHNAMES: Pathnames to headers to verify.
# 
# Cache veriables will be set that are named after the headers:
# <header_name>_IS_STANDALONE
macro(verify_headers_are_self_sufficient)
    set(OPTIONS "")
    set(ONE_VALUE_ARGUMENTS OFFSET_PATHNAME FLAGS)
    set(MULTI_VALUE_ARGUMENTS INCLUDES LIBRARIES HEADER_PATHNAMES)
    cmake_parse_arguments(VERIFY_HEADERS "${OPTIONS}" "${ONE_VALUE_ARGUMENTS}"
        "${MULTI_VALUE_ARGUMENTS}" ${ARGN})

    if(VERIFY_HEADERS_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "Macro called with unrecognized arguments: "
            "${VERIFY_HEADERS_UNPARSED_ARGUMENTS}"
        )
    endif()

    set(CMAKE_REQUIRED_FLAGS "${VERIFY_HEADERS_FLAGS}")
    # set(CMAKE_REQUIRED_DEFINITIONS xxx)
    set(CMAKE_REQUIRED_INCLUDES ${VERIFY_HEADERS_INCLUDES})
    set(CMAKE_REQUIRED_LIBRARIES ${VERIFY_HEADERS_LIBRARIES})

    foreach(HEADER_PATHNAME ${VERIFY_HEADERS_HEADER_PATHNAMES})
        string(REPLACE ${VERIFY_HEADERS_OFFSET_PATHNAME} "" HEADER_NAME
            ${HEADER_PATHNAME})

        # Create variable name that contains the name of the header being
        # checked and is a valid macro name. It is passed to the compiler:
        # -D${VARIABLE_NAME}. That meanѕ that some characters cannot be in
        # the name.
        set(VARIABLE_NAME ${HEADER_NAME})
        string(REPLACE /  _ VARIABLE_NAME ${VARIABLE_NAME})
        string(REPLACE \\ _ VARIABLE_NAME ${VARIABLE_NAME})
        string(REPLACE .  _ VARIABLE_NAME ${VARIABLE_NAME})

        # - Include the header twice to see whether the '#pragma once' is in
        #   place.
        # - Compile a dummy main to see whether the header includes everything
        #   it uses.
        check_cxx_source_compiles("
            #include \"${HEADER_NAME}\"
            #include \"${HEADER_NAME}\"
            int main(int /* argc */, char** /* argv */) {
              return 0;
            }"
            ${VARIABLE_NAME})

        if(NOT ${VARIABLE_NAME})
            message(FATAL_ERROR
                "Header ${HEADER_NAME} is not self-sufficient. "
                "Inspect CMakeFiles/{CMakeError.log,CMakeOutput.log}."
            )
        endif()
    endforeach()
endmacro()


# TODO Can we somehow configure the extension to end up in bin/python instead
# TODO of bin? Currently we cannot have a dll and a python extension
# TODO both named bla. On Windows the import lib of the python extension will
# TODO conflict with the import lib of the dll.
macro(configure_python_extension
        EXTENTION_TARGET
        EXTENSION_NAME)
    set_target_properties(${EXTENTION_TARGET}
        PROPERTIES
            OUTPUT_NAME "${EXTENSION_NAME}"
    )

    # Configure suffix and prefix, depending on the Python OS conventions.
    set_target_properties(${EXTENTION_TARGET}
        PROPERTIES
            PREFIX ""
    )

    if(WIN32)
        set_target_properties(${EXTENTION_TARGET}
            PROPERTIES
                DEBUG_POSTFIX "_d"
                SUFFIX ".pyd"
        )
    else(WIN32)
        set_target_properties(${EXTENTION_TARGET}
            PROPERTIES
                SUFFIX ".so"
        )
    endif(WIN32)
endmacro()


# Add a test target.
# Also configures the environment to point to the location of shared libs.
# The idea of this is to keep the dev's shell as clean as possible. Use
# ctest command to run unit tests.
# SCOPE: Some prefix. Often the lib name of the lib being tested.
# NAME : Name of test module, without extension.
# LINK_LIBRARIES: Libraries to link against.
# DEPENDENCIES: Targets this test target depends on.
macro(add_unit_test2)
    set(OPTIONS "")
    set(ONE_VALUE_ARGUMENTS SCOPE NAME)
    set(MULTI_VALUE_ARGUMENTS LINK_LIBRARIES DEPENDENCIES)

    cmake_parse_arguments(ADD_UNIT_TEST "${OPTIONS}" "${ONE_VALUE_ARGUMENTS}"
        "${MULTI_VALUE_ARGUMENTS}" ${ARGN})

    if(ADD_UNIT_TEST_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "Macro called with unrecognized arguments: "
            "${ADD_UNIT_TEST_UNPARSED_ARGUMENTS}"
        )
    endif()

    set(TEST_MODULE_NAME ${ADD_UNIT_TEST_NAME})
    set(TEST_EXE_NAME ${ADD_UNIT_TEST_SCOPE}_${TEST_MODULE_NAME})

    add_executable(${TEST_EXE_NAME} ${TEST_MODULE_NAME})
    target_link_libraries(${TEST_EXE_NAME}
        ${ADD_UNIT_TEST_LINK_LIBRARIES}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        stdc++)
    add_test(NAME ${TEST_EXE_NAME}
        # catch_system_errors: Prevent UTF to detect system errors. This
        #     messes things up when doing system calls to Python unit tests.
        #     See also: http://lists.boost.org/boost-users/2009/12/55048.php
        COMMAND ${TEST_EXE_NAME} --catch_system_errors=no)

    if(ADD_UNIT_TEST_DEPENDENCIES)
        ADD_DEPENDENCIES(${TEST_EXE_NAME} ${ADD_UNIT_TEST_DEPENDENCIES})
    endif()

    # Maybe add ${EXECUTABLE_OUTPUT_PATH} in the future. If needed.
    set(PATH_LIST $ENV{PATH})
    list(INSERT PATH_LIST 0 ${Boost_LIBRARY_DIRS})
    set(PATH_STRING "${PATH_LIST}")

    if(${host_system_name} STREQUAL "windows")
        string(REPLACE "\\" "/" PATH_STRING "${PATH_STRING}")
        string(REPLACE ";" "\\;" PATH_STRING "${PATH_STRING}")
    else()
        string(REPLACE ";" ":" PATH_STRING "${PATH_STRING}")
    endif()

    set_property(TEST ${TEST_EXE_NAME}
        PROPERTY ENVIRONMENT "PATH=${PATH_STRING}")
endmacro()


# Copy Python test modules from current source directory to current binary
# directory. For each module a custom command is created so editing a test
# module in the source directory will trigger a copy to the binary directory.
# Also, a custom target is defined that depends on all copied test modules.
# If you let another target depend on this custom target, then all copied
# test modules will always be up to date before building the other target.
# TARGET: Name of custom target to add.
macro(copy_python_unit_test_modules)
    set(OPTIONS "")
    set(ONE_VALUE_ARGUMENTS TARGET)
    set(MULTI_VALUE_ARGUMENTS "")

    cmake_parse_arguments(COPY_MODULES "${OPTIONS}" "${ONE_VALUE_ARGUMENTS}"
        "${MULTI_VALUE_ARGUMENTS}" ${ARGN})

    if(COPY_MODULES_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "Macro called with unrecognized arguments: "
            "${COPY_MODULES_UNPARSED_ARGUMENTS}"
        )
    endif()

    file(GLOB PYTHON_UNIT_TEST_MODULES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*_test.py")

    foreach(MODULE ${PYTHON_UNIT_TEST_MODULES})
        set(PYTHON_UNIT_TEST_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE})
        set(COPIED_PYTHON_UNIT_TEST_MODULE
            ${CMAKE_CURRENT_BINARY_DIR}/${MODULE})
        add_custom_command(
            OUTPUT ${COPIED_PYTHON_UNIT_TEST_MODULE}
            DEPENDS ${PYTHON_UNIT_TEST_MODULE}
            COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_UNIT_TEST_MODULE}
                ${COPIED_PYTHON_UNIT_TEST_MODULE}
        )
        list(APPEND COPIED_PYTHON_UNIT_TEST_MODULES
            ${COPIED_PYTHON_UNIT_TEST_MODULE})
    endforeach()

    add_custom_target(${COPY_MODULES_TARGET}
        DEPENDS ${COPIED_PYTHON_UNIT_TEST_MODULES})
endmacro()


macro(copy_python_modules)
    set(OPTIONS "")
    set(ONE_VALUE_ARGUMENTS TARGET TARGET_DIRECTORY)
    set(MULTI_VALUE_ARGUMENTS "")

    cmake_parse_arguments(COPY_MODULES "${OPTIONS}" "${ONE_VALUE_ARGUMENTS}"
        "${MULTI_VALUE_ARGUMENTS}" ${ARGN})

    if(COPY_MODULES_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "Macro called with unrecognized arguments: "
            "${COPY_MODULES_UNPARSED_ARGUMENTS}"
        )
    endif()

    file(GLOB PYTHON_MODULES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")

    foreach(MODULE ${PYTHON_MODULES})
        set(PYTHON_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE})
        set(COPIED_PYTHON_MODULE ${COPY_MODULES_TARGET_DIRECTORY}/${MODULE})
        add_custom_command(
            OUTPUT ${COPIED_PYTHON_MODULE}
            DEPENDS ${PYTHON_MODULE}
            COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_MODULE}
                ${COPIED_PYTHON_MODULE}
        )
        list(APPEND COPIED_PYTHON_MODULES ${COPIED_PYTHON_MODULE})
    endforeach()

    add_custom_target(${COPY_MODULES_TARGET}
        DEPENDS ${COPIED_PYTHON_MODULES})
endmacro()
