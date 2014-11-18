# Peacock is a project for building external software. It can be used to
# build the Boost libraries on all kinds of platform, for example. A
# PEACOCK_PREFIX CMake variable or environment variable can be set to
# point us to the root of the platform-specific files. By adding the
# current platform string to this prefix, we end up at the root of the
# header files and libraries.
# See also: https://github.com/geoneric/peacock

# If the PEACOCK_PREFIX CMake variable is not set, but an environment
# variable with that name is, then copy it to a CMake variable. This way
# the CMake variable takes precedence.
IF((NOT PEACOCK_PREFIX) AND (DEFINED ENV{PEACOCK_PREFIX}))
    SET(PEACOCK_PREFIX $ENV{PEACOCK_PREFIX})
ENDIF()

IF(PEACOCK_PREFIX)
    # # if cross compiling:
    # SET(CMAKE_FIND_ROOT_PATH
    #     ${PEACOCK_PREFIX}/${peacock_target_platform})
    # else:
    SET(CMAKE_PREFIX_PATH
        ${PEACOCK_PREFIX}/${peacock_target_platform}
        ${CMAKE_PREFIX_PATH}
    )
ENDIF()


# Configure and find packages, configure project. ------------------------------
IF(FERN_BOOST_REQUIRED)
    SET(Boost_USE_STATIC_LIBS OFF)
    SET(Boost_USE_STATIC_RUNTIME OFF)
    ADD_DEFINITIONS(
        # Use dynamic libraries.
        -DBOOST_ALL_DYN_LINK
        # Prevent auto-linking.
        -DBOOST_ALL_NO_LIB

        # # No deprecated features.
        # -DBOOST_FILESYSTEM_NO_DEPRECATED

        # -DBOOST_CHRONO_DONT_PROVIDE_HYBRID_ERROR_HANDLING
        # -DBOOST_CHRONO_HEADER_ONLY
    )
    SET(CMAKE_CXX_FLAGS_RELEASE
        # Disable range checks in release builds.
        "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_DISABLE_ASSERTS"
    )
    FIND_PACKAGE(Boost REQUIRED
        COMPONENTS ${FERN_REQUIRED_BOOST_COMPONENTS})
    INCLUDE_DIRECTORIES(
        SYSTEM
        ${Boost_INCLUDE_DIRS}
    )
    LIST(APPEND FERN_EXTERNAL_LIBRARIES
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
    )
ENDIF()
### IF(FERN_CYTHON_REQUIRED)
###     FIND_PACKAGE(Cython REQUIRED)
###     INCLUDE(UseCython)
### ENDIF()
FIND_PACKAGE(Doxygen)
IF(FERN_EXPAT_REQUIRED)
    FIND_PACKAGE(EXPAT REQUIRED)
    INCLUDE_DIRECTORIES(
        SYSTEM
        ${EXPAT_INCLUDE_DIRS}
    )
ENDIF()
IF(FERN_GDAL_REQUIRED)
    FIND_PACKAGE(GDAL REQUIRED)
    INCLUDE_DIRECTORIES(
        SYSTEM
        ${GDAL_INCLUDE_DIRS}
    )
    LIST(APPEND FERN_EXTERNAL_LIBRARIES
        ${GDAL_LIBRARIES}
    )
ENDIF()
IF(FERN_HDF5_REQUIRED)
    SET(HDF5_USE_STATIC_LIBRARIES OFF)
    FIND_PACKAGE(HDF5 REQUIRED
        COMPONENTS C CXX HL)
    INCLUDE_DIRECTORIES(
        SYSTEM
        ${HDF5_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}/cpp
    )
    LIST(APPEND FERN_EXTERNAL_LIBRARIES
        ${HDF5_LIBRARIES}
    )
ENDIF()
IF(FERN_LOKI_REQUIRED)
    FIND_PACKAGE(Loki REQUIRED)
ENDIF()
IF(FERN_NETCDF_REQUIRED)
    FIND_PACKAGE(NetCDF REQUIRED)
ENDIF()
IF(FERN_NUMPY_REQUIRED)
    FIND_PACKAGE(NumPy REQUIRED)
    INCLUDE_DIRECTORIES(
        SYSTEM
        ${NUMPY_INCLUDE_DIRS}
    )
    # http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html
    ADD_DEFINITIONS(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
ENDIF()
IF(FERN_PYTHON_LIBS_REQUIRED)
    SET(Python_ADDITIONAL_VERSIONS "2.7")
    FIND_PACKAGE(PythonLibs REQUIRED)
    INCLUDE_DIRECTORIES(
        SYSTEM
        ${PYTHON_INCLUDE_DIRS}
    )
    LIST(APPEND FERN_EXTERNAL_LIBRARIES
        ${PYTHON_LIBRARIES}
    )
ENDIF()
IF(FERN_READLINE_REQUIRED)
    FIND_PACKAGE(Readline REQUIRED)
    INCLUDE_DIRECTORIES(
        SYSTEM
        ${READLINE_INCLUDE_DIR}
    )
ENDIF()
IF(FERN_SWIG_REQUIRED)
    FIND_PACKAGE(SWIG REQUIRED)
ENDIF()
IF(FERN_XSD_REQUIRED)
    FIND_PACKAGE(XSD REQUIRED)
    INCLUDE_DIRECTORIES(
        SYSTEM
        ${XSD_INCLUDE_DIRS}
    )
ENDIF()
