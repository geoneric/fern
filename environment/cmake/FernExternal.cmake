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
if((NOT PEACOCK_PREFIX) AND (DEFINED ENV{PEACOCK_PREFIX}))
    set(PEACOCK_PREFIX $ENV{PEACOCK_PREFIX})
endif()

if(PEACOCK_PREFIX)
    # # if cross compiling:
    # set(CMAKE_FIND_ROOT_PATH
    #     ${PEACOCK_PREFIX}/${peacock_target_platform})
    # else:
    set(CMAKE_PREFIX_PATH
        ${PEACOCK_PREFIX}/${peacock_target_platform}
        ${CMAKE_PREFIX_PATH}
    )
endif()


# Configure and find packages, configure project. ------------------------------
if(FERN_BOOST_REQUIRED)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_STATIC_RUNTIME OFF)
    add_definitions(
        # Use dynamic libraries.
        -DBOOST_ALL_DYN_LINK
        # Prevent auto-linking.
        -DBOOST_ALL_NO_LIB

        # # No deprecated features.
        # -DBOOST_FILESYSTEM_NO_DEPRECATED

        # -DBOOST_CHRONO_DONT_PROVIDE_HYBRID_ERROR_HANDLING
        # -DBOOST_CHRONO_HEADER_ONLY
    )
    set(CMAKE_CXX_FLAGS_RELEASE
        # Disable range checks in release builds.
        "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_DISABLE_ASSERTS"
    )
    list(REMOVE_DUPLICATES FERN_REQUIRED_BOOST_COMPONENTS)
    find_package(Boost REQUIRED
        COMPONENTS ${FERN_REQUIRED_BOOST_COMPONENTS})
    include_directories(
        SYSTEM
        ${Boost_INCLUDE_DIRS}
    )
    list(APPEND FERN_EXTERNAL_LIBRARIES
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
    )
endif()
if(FERN_BUILD_DOCUMENTATION)
    find_package(Doxygen REQUIRED)
endif()
if(FERN_EXPAT_REQUIRED)
    find_package(EXPAT REQUIRED)
    include_directories(
        SYSTEM
        ${EXPAT_INCLUDE_DIRS}
    )
endif()
if(FERN_GDAL_REQUIRED)
    find_package(GDAL REQUIRED)
    include_directories(
        SYSTEM
        ${GDAL_INCLUDE_DIRS}
    )
    list(APPEND FERN_EXTERNAL_LIBRARIES
        ${GDAL_LIBRARIES}
    )
    find_program(GDAL_TRANSLATE gdal_translate
        HINTS ${GDAL_INCLUDE_DIR}/../bin
    )
endif()
if(FERN_HDF5_REQUIRED)
    set(HDF5_USE_STATIC_LIBRARIES OFF)
    find_package(HDF5 REQUIRED
        COMPONENTS C CXX HL)
    include_directories(
        SYSTEM
        ${HDF5_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}/cpp
    )
    list(APPEND FERN_EXTERNAL_LIBRARIES
        ${HDF5_LIBRARIES}
    )
endif()
if(FERN_HPX_REQUIRED)
    # TODO Set HPX_DIR to dir containing HPXConfig.cmake.
    # See lib/cmake/hpx/HPXTargets.cmake for names of HPX targets to link
    # against.

    # HPX updates CMAKE_CXX_FLAGS (adds -std=c++11, see HPXConfig.cmake).
    # We want to do this ourselves.
    set(_flags ${CMAKE_CXX_FLAGS})
    find_package(HPX REQUIRED)
    set(CMAKE_CXX_FLAGS ${_flags})
    include_directories(${HPX_INCLUDE_DIRS})
endif()
if(FERN_LOKI_REQUIRED)
    find_package(Loki REQUIRED)
endif()
if(FERN_NETCDF_REQUIRED)
    find_package(NetCDF REQUIRED)
    include_directories(
        SYSTEM
        ${NetCDF_INCLUDE_DIRECTORIES}
    )
    find_program(NCGEN ncgen
        HINTS ${NetCDF_INCLUDE_DIRECTORIES}/../bin
    )
    list(APPEND FERN_EXTERNAL_LIBRARIES
        ${NetCDF_LIBRARIES}
    )
    message(STATUS "Found NetCDF:")
    message(STATUS "  includes : ${NetCDF_INCLUDE_DIRECTORIES}")
    message(STATUS "  C-library: ${NetCDF_C_LIBRARY}")
endif()
if(FERN_NUMPY_REQUIRED)
    find_package(NumPy REQUIRED)
    include_directories(
        SYSTEM
        ${NUMPY_INCLUDE_DIRS}
    )
    # http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html
    add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
endif()
if(FERN_PYTHON_INTERP_REQUIRED)
    set(Python_ADDITIONAL_VERSIONS "2.7")
    find_package(PythonInterp REQUIRED)
endif()
if(FERN_PYTHON_LIBS_REQUIRED)
    set(Python_ADDITIONAL_VERSIONS "2.7")
    find_package(PythonLibs REQUIRED)
    include_directories(
        SYSTEM
        ${PYTHON_INCLUDE_DIRS}
    )
    list(APPEND FERN_EXTERNAL_LIBRARIES
        ${PYTHON_LIBRARIES}
    )
endif()
if(FERN_READLINE_REQUIRED)
    find_package(Readline REQUIRED)
    include_directories(
        SYSTEM
        ${READLINE_INCLUDE_DIR}
    )
endif()
if(FERN_SWIG_REQUIRED)
    find_package(SWIG REQUIRED)
endif()
if(FERN_XSD_REQUIRED)
    find_package(XSD REQUIRED)
    include_directories(
        SYSTEM
        ${XSD_INCLUDE_DIRS}
    )
endif()
