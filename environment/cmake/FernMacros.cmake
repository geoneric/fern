INCLUDE(CheckCXXSourceRuns)


MACRO(ADD_PARSER_GENERATION_COMMAND
        BASENAME)
    CONFIGURE_FILE(
        ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.map.in
        ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.map
    )

    ADD_CUSTOM_COMMAND(
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
ENDMACRO()


# Targets can be added conditionally. When Fern components are selected during
# configuration, certain variables are set that tell us whether or not to
# include a target or group of targets. See FernConfiguration.cmake.
# TARGET_NAME   : Name of target, snake_case.
# DIRECTORY_NAME: Name of subdirectory containing the target.
MACRO(ADD_TARGET_CONDITIONALLY
        TARGET_NAME
        DIRECTORY_NAME)
    # Determine name of variable to check.
    STRING(TOUPPER ${TARGET_NAME} VARIABLE_NAME)
    SET(VARIABLE_NAME "FERN_FERN_${VARIABLE_NAME}_REQUIRED")

    # Make sure it is defined.
    IF(NOT DEFINED ${VARIABLE_NAME})
        MESSAGE(SEND_ERROR "Variable ${VARIABLE_NAME} is not defined")
    ENDIF()

    # Evaluate the variable and include target if result is true.
    IF(${VARIABLE_NAME})
        ADD_SUBDIRECTORY(${DIRECTORY_NAME})
    ENDIF()
ENDMACRO()


# Create a static library and an object library.
# BASENAME: Name of static library to create. The object library will be 
#           named ${BASENAME}_objects.
# SOURCES : Sources that are part of the libraries. Any argument that comes
#           after the BASENAME is treated as a source.
MACRO(ADD_LIBRARY_AND_OBJECT_LIBRARY
        BASENAME)
    SET(SOURCES ${ARGN})
    ADD_LIBRARY(${BASENAME}
        ${SOURCES}
    )
    ADD_LIBRARY(${BASENAME}_objects OBJECT
        ${SOURCES}
    )
ENDMACRO()


# Verify that headers are self-sufficient: they include the headers they need.
# OFFSET_PATHNAME : Pathname to root of headers.
# INCLUDES        : Pathnames where included headers can be found.
# LIBRARIES       : Libraries to link.
# FLAGS           : Compiler flags.
# HEADER_PATHNAMES: Pathnames to headers to verify.
# 
# Cache veriables will be set that are named after the headers:
# <header_name>_IS_STANDALONE
MACRO(VERIFY_HEADERS_ARE_SELF_SUFFICIENT)
    SET(OPTIONS "")
    SET(ONE_VALUE_ARGUMENTS OFFSET_PATHNAME FLAGS)
    SET(MULTI_VALUE_ARGUMENTS INCLUDES LIBRARIES HEADER_PATHNAMES)
    CMAKE_PARSE_ARGUMENTS(VERIFY_HEADERS "${OPTIONS}" "${ONE_VALUE_ARGUMENTS}"
        "${MULTI_VALUE_ARGUMENTS}" ${ARGN})

    IF(VERIFY_HEADERS_UNPARSED_ARGUMENTS)
        MESSAGE(FATAL_ERROR
            "Macro called with unrecognized arguments: "
            "${VERIFY_HEADERS_UNPARSED_ARGUMENTS}"
        )
    ENDIF()

    SET(CMAKE_REQUIRED_FLAGS "${VERIFY_HEADERS_FLAGS}")
    # SET(CMAKE_REQUIRED_DEFINITIONS xxx)
    SET(CMAKE_REQUIRED_INCLUDES ${VERIFY_HEADERS_INCLUDES})
    SET(CMAKE_REQUIRED_LIBRARIES ${VERIFY_HEADERS_LIBRARIES})

    FOREACH(HEADER_PATHNAME ${VERIFY_HEADERS_HEADER_PATHNAMES})
        STRING(REPLACE ${VERIFY_HEADERS_OFFSET_PATHNAME} "" HEADER_NAME
            ${HEADER_PATHNAME})

        # Create variable name that contains the name of the header being
        # checked and is a valid macro name. It is passed to the compiler:
        # -D${VARIABLE_NAME}. That meanѕ that some characters cannot be in
        # the name.
        SET(VARIABLE_NAME ${HEADER_NAME})
        STRING(REPLACE /  _ VARIABLE_NAME ${VARIABLE_NAME})
        STRING(REPLACE \\ _ VARIABLE_NAME ${VARIABLE_NAME})
        STRING(REPLACE .  _ VARIABLE_NAME ${VARIABLE_NAME})

        # - Include the header twice to see whether the '#pragma once' is in
        #   place.
        # - Compile a dummy main to see whether the header includes everything
        #   it uses.
        CHECK_CXX_SOURCE_COMPILES("
            #include \"${HEADER_NAME}\"
            #include \"${HEADER_NAME}\"
            int main(int /* argc */, char** /* argv */) {
              return 0;
            }"
            ${VARIABLE_NAME})

        IF(NOT ${VARIABLE_NAME})
            MESSAGE(FATAL_ERROR
                "Header ${HEADER_NAME} is not self-sufficient. "
                "Inspect CMakeFiles/{CMakeError.log,CMakeOutput.log}."
            )
        ENDIF()
    ENDFOREACH()
ENDMACRO()


# TODO Can we somehow configure the extension to end up in bin/python instead
# TODO of bin? Currently we cannot have a dll and a python extension
# TODO both named bla. On Windows the import lib of the python extension will
# TODO conflict with the import lib of the dll.
MACRO(CONFIGURE_PYTHON_EXTENSION
        EXTENTION_TARGET
        EXTENSION_NAME)
    SET_TARGET_PROPERTIES(${EXTENTION_TARGET}
        PROPERTIES
            OUTPUT_NAME "${EXTENSION_NAME}"
    )

    # Configure suffix and prefix, depending on the Python OS conventions.
    SET_TARGET_PROPERTIES(${EXTENTION_TARGET}
        PROPERTIES
            PREFIX ""
    )

    IF(WIN32)
        SET_TARGET_PROPERTIES(${EXTENTION_TARGET}
            PROPERTIES
                DEBUG_POSTFIX "_d"
                SUFFIX ".pyd"
        )
    ELSE(WIN32)
        SET_TARGET_PROPERTIES(${EXTENTION_TARGET}
            PROPERTIES
                SUFFIX ".so"
        )
    ENDIF(WIN32)
ENDMACRO()


# Add a test target.
# Also configures the environment to point to the location of shared libs.
# The idea of this is to keep the dev's shell as clean as possible. Use
# ctest command to run unit tests.
# SCOPE: Some prefix. Often the lib name of the lib being tested.
# NAME : Name of test module, without extension.
# LINK_LIBRARIES: Libraries to link against.
# DEPENDENCIES: Targets this test target depends on.
MACRO(ADD_UNIT_TEST2)
    SET(OPTIONS "")
    SET(ONE_VALUE_ARGUMENTS SCOPE NAME)
    SET(MULTI_VALUE_ARGUMENTS LINK_LIBRARIES DEPENDENCIES)

    CMAKE_PARSE_ARGUMENTS(ADD_UNIT_TEST "${OPTIONS}" "${ONE_VALUE_ARGUMENTS}"
        "${MULTI_VALUE_ARGUMENTS}" ${ARGN})

    IF(ADD_UNIT_TEST_UNPARSED_ARGUMENTS)
        MESSAGE(FATAL_ERROR
            "Macro called with unrecognized arguments: "
            "${ADD_UNIT_TEST_UNPARSED_ARGUMENTS}"
        )
    ENDIF()

    SET(TEST_MODULE_NAME ${ADD_UNIT_TEST_NAME})
    SET(TEST_EXE_NAME ${ADD_UNIT_TEST_SCOPE}_${TEST_MODULE_NAME})

    ADD_EXECUTABLE(${TEST_EXE_NAME} ${TEST_MODULE_NAME})
    TARGET_LINK_LIBRARIES(${TEST_EXE_NAME}
        ${ADD_UNIT_TEST_LINK_LIBRARIES}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        stdc++)
    ADD_TEST(NAME ${TEST_EXE_NAME}
        # catch_system_errors: Prevent UTF to detect system errors. This
        #     messes things up when doing system calls to Python unit tests.
        #     See also: http://lists.boost.org/boost-users/2009/12/55048.php
        COMMAND ${TEST_EXE_NAME} --catch_system_errors=no)

    IF(ADD_UNIT_TEST_DEPENDENCIES)
        ADD_DEPENDENCIES(${TEST_EXE_NAME} ${ADD_UNIT_TEST_DEPENDENCIES})
    ENDIF()

    # Maybe add ${EXECUTABLE_OUTPUT_PATH} in the future. If needed.
    SET(PATH_LIST $ENV{PATH})
    LIST(INSERT PATH_LIST 0 ${Boost_LIBRARY_DIRS})
    SET(PATH_STRING "${PATH_LIST}")

    IF(${host_system_name} STREQUAL "windows")
        STRING(REPLACE "\\" "/" PATH_STRING "${PATH_STRING}")
        STRING(REPLACE ";" "\\;" PATH_STRING "${PATH_STRING}")
    ELSE()
        STRING(REPLACE ";" ":" PATH_STRING "${PATH_STRING}")
    ENDIF()

    SET_PROPERTY(TEST ${TEST_EXE_NAME}
        PROPERTY ENVIRONMENT "PATH=${PATH_STRING}")
ENDMACRO()


# Copy Python test modules from current source directory to current binary
# directory. For each module a custom command is created so editing a test
# module in the source directory will trigger a copy to the binary directory.
# Also, a custom target is defined that depends on all copied test modules.
# If you let another target depend on this custom target, then all copied
# test modules will always be up to date before building the other target.
# TARGET: Name of custom target to add.
MACRO(COPY_PYTHON_UNIT_TEST_MODULES)
    SET(OPTIONS "")
    SET(ONE_VALUE_ARGUMENTS TARGET)
    SET(MULTI_VALUE_ARGUMENTS "")

    CMAKE_PARSE_ARGUMENTS(COPY_MODULES "${OPTIONS}" "${ONE_VALUE_ARGUMENTS}"
        "${MULTI_VALUE_ARGUMENTS}" ${ARGN})

    IF(COPY_MODULES_UNPARSED_ARGUMENTS)
        MESSAGE(FATAL_ERROR
            "Macro called with unrecognized arguments: "
            "${COPY_MODULES_UNPARSED_ARGUMENTS}"
        )
    ENDIF()

    FILE(GLOB PYTHON_UNIT_TEST_MODULES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*_test.py")

    FOREACH(MODULE ${PYTHON_UNIT_TEST_MODULES})
        SET(PYTHON_UNIT_TEST_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE})
        SET(COPIED_PYTHON_UNIT_TEST_MODULE
            ${CMAKE_CURRENT_BINARY_DIR}/${MODULE})
        ADD_CUSTOM_COMMAND(
            OUTPUT ${COPIED_PYTHON_UNIT_TEST_MODULE}
            DEPENDS ${PYTHON_UNIT_TEST_MODULE}
            COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_UNIT_TEST_MODULE}
                ${COPIED_PYTHON_UNIT_TEST_MODULE}
        )
        LIST(APPEND COPIED_PYTHON_UNIT_TEST_MODULES
            ${COPIED_PYTHON_UNIT_TEST_MODULE})
    ENDFOREACH()

    ADD_CUSTOM_TARGET(${COPY_MODULES_TARGET}
        DEPENDS ${COPIED_PYTHON_UNIT_TEST_MODULES})
ENDMACRO()
