# Options for selecting the modules to build.
# FERN_BUILD_<module>

# Options for selecting features.
# FERN_WITH_<feature>

option(FERN_BUILD_ALL "Build everything, except for documentation and tests"
    FALSE)
option(FERN_WITH_ALL "Support all features" FALSE)

option(FERN_BUILD_ALGORITHM "Build Fern.Algorithm module" FALSE)

option(FERN_BUILD_IO "Build Fern.IO module" FALSE)
option(FERN_IO_WITH_GDAL "Add support for GDAL" FALSE)
option(FERN_IO_WITH_GPX "Add support for GPX" FALSE)
option(FERN_IO_WITH_HDF5 "Add support for HDF5" FALSE)
option(FERN_IO_WITH_NETCDF "Add support for NetCDF" FALSE)
# option(FERN_IO_WITH_SVG "Add support for SVG" FALSE)

option(FERN_BUILD_LANGUAGE "Build Fern.Language module" FALSE)
option(FERN_LANGUAGE_WITH_INTERPRETER "Add support for executing Fern scripts"
    FALSE)
option(FERN_LANGUAGE_WITH_COMPILER "Add support for compiling Fern scripts"
    FALSE)

option(FERN_BUILD_PYTHON "Build Fern.Python module" FALSE)

option(FERN_BUILD_DOCUMENTATION "Build documentation" FALSE)
option(FERN_BUILD_TEST "Build tests" FALSE)


# TODO Make this an option. Allow this to be changed per library.
set(FERN_ALL_DYN_LINK FALSE)
if(NOT WIN32)
    set(FERN_ALL_DYN_LINK TRUE)
endif()

if(FERN_ALL_DYN_LINK)
    set(FERN_ALGORITHM_DYN_LINK TRUE)
    set(FERN_CORE_DYN_LINK TRUE)
    set(FERN_FEATURE_DYN_LINK TRUE)
else()
    set(FERN_ALGORITHM_DYN_LINK FALSE)
    set(FERN_CORE_DYN_LINK FALSE)
    set(FERN_FEATURE_DYN_LINK FALSE)
endif()


# Some modules require the build of other modules and support for certain
# features.
if(FERN_BUILD_ALL)
    set(FERN_BUILD_ALGORITHM TRUE)
    set(FERN_BUILD_IO TRUE)
    set(FERN_BUILD_LANGUAGE TRUE)
    set(FERN_BUILD_PYTHON TRUE)
endif()

if(FERN_BUILD_LANGUAGE)
    set(FERN_BUILD_ALGORITHM TRUE)
    set(FERN_BUILD_IO TRUE)
    set(FERN_IO_WITH_HDF5 TRUE)  # Fern data format is in HDF5.
    set(FERN_IO_WITH_GDAL TRUE)
endif()

if(FERN_BUILD_PYTHON)
    set(FERN_BUILD_ALGORITHM TRUE)
    set(FERN_BUILD_IO TRUE)
    set(FERN_IO_WITH_GDAL TRUE)
endif()


# Some features require the selection of other features.
if(FERN_WITH_ALL)
    set(FERN_IO_WITH_ALL TRUE)
    set(FERN_LANGUAGE_WITH_ALL TRUE)
endif()

if(FERN_IO_WITH_ALL)
    set(FERN_IO_WITH_GDAL TRUE)
    set(FERN_IO_WITH_GPX TRUE)
    set(FERN_IO_WITH_HDF5 TRUE)
    set(FERN_IO_WITH_NETCDF TRUE)
    # set(FERN_IO_WITH_SVG TRUE)
endif()

if(FERN_IO_WITH_NETCDF)
    set(FERN_IO_WITH_HDF5 TRUE)
endif()

if(FERN_LANGUAGE_WITH_ALL)
    set(FERN_LANGUAGE_WITH_INTERPRETER TRUE)
    set(FERN_LANGUAGE_WITH_COMPILER TRUE)
endif()


# Depending on the selection of Fern software made during configuration,
# some third party software is required and other isn't.
if(FERN_BUILD_ALGORITHM)
    # Required third party software.
    set(DEVBASE_BOOST_REQUIRED TRUE)
    list(APPEND DEVBASE_REQUIRED_BOOST_COMPONENTS
        filesystem system timer)

    # Required Fern targets.
    set(FERN_FERN_ALGORITHM_REQUIRED TRUE)
    set(FERN_FERN_CORE_REQUIRED TRUE)
    set(FERN_FERN_EXAMPLE_REQUIRED TRUE)
    set(FERN_FERN_FEATURE_REQUIRED TRUE)
endif()


if(FERN_BUILD_DOCUMENTATION)
    set(DEVBASE_DOXYGEN_REQUIRED TRUE)
endif()


# if(FERN_HPX)
#     # Required third party software.
#     set(DEVBASE_BOOST_REQUIRED TRUE)
#     list(APPEND DEVBASE_REQUIRED_BOOST_COMPONENTS
#         date_time program_options regex serialization thread chrono)
#     set(FERN_HPX_REQUIRED TRUE)
# 
#     # Required Fern targets.
#     set(FERN_FERN_HPX_REQUIRED TRUE)
# endif()


if(FERN_BUILD_IO)
    # Required third party software.
    set(DEVBASE_BOOST_REQUIRED TRUE)
    list(APPEND DEVBASE_REQUIRED_BOOST_COMPONENTS
        filesystem system timer)
    if(FERN_IO_WITH_GDAL)
        set(DEVBASE_GDAL_REQUIRED TRUE)
    endif()
    if(FERN_IO_WITH_GPX)
        set(DEVBASE_XERCES_REQUIRED TRUE)
        set(DEVBASE_XSD_REQUIRED TRUE)
    endif()
    if(FERN_IO_WITH_HDF5)
        set(DEVBASE_HDF5_REQUIRED TRUE)
        set(DEVBASE_REQUIRED_HDF5_COMPONENTS C HL)
    endif()
    if(FERN_IO_WITH_NETCDF)
        set(DEVBASE_NETCDF_REQUIRED TRUE)
    endif()

    # Required Fern targets.
    set(FERN_FERN_CORE_REQUIRED TRUE)
    if(FERN_BUILD_TEST)
        set(FERN_FERN_FEATURE_REQUIRED TRUE)
    endif()
    set(FERN_FERN_IO_REQUIRED TRUE)
endif()


if(FERN_BUILD_LANGUAGE)
    # Required third party software.
    set(DEVBASE_BOOST_REQUIRED TRUE)
    list(APPEND DEVBASE_REQUIRED_BOOST_COMPONENTS
        filesystem system timer)
    set(DEVBASE_EXPAT_REQUIRED TRUE)
    set(DEVBASE_LOKI_REQUIRED TRUE)
    set(DEVBASE_PYTHON_LIBS_REQUIRED TRUE)
    set(DEVBASE_REQUIRED_PYTHON_VERSION 2.7)
    set(DEVBASE_READLINE_REQUIRED TRUE)
    set(DEVBASE_XSD_REQUIRED TRUE)

    # Required Fern targets.
    set(FERN_FERN_CORE_REQUIRED TRUE)
    set(FERN_FERN_FEATURE_REQUIRED TRUE)

    set(FERN_FERN_LANGUAGE_AST_REQUIRED TRUE)
    set(FERN_FERN_LANGUAGE_BACK_END_REQUIRED TRUE)
    if(FERN_LANGUAGE_WITH_COMPILER OR FERN_LANGUAGE_WITH_INTERPRETER)
        set(FERN_FERN_LANGUAGE_COMMAND_REQUIRED TRUE)
    endif()
    if(FERN_LANGUAGE_WITH_COMPILER)
        set(FERN_FERN_LANGUAGE_COMPILER_REQUIRED TRUE)
    endif()
    set(FERN_FERN_LANGUAGE_CORE_REQUIRED TRUE)
    set(FERN_FERN_LANGUAGE_EXECUTOR_REQUIRED TRUE)
    if(FERN_LANGUAGE_WITH_INTERPRETER)
        set(FERN_FERN_LANGUAGE_INTERPRETER_REQUIRED TRUE)
    endif()
    set(FERN_FERN_LANGUAGE_IO_REQUIRED TRUE)
    set(FERN_FERN_LANGUAGE_FEATURE_REQUIRED TRUE)
    set(FERN_FERN_LANGUAGE_OPERATION_REQUIRED TRUE)
    set(FERN_FERN_LANGUAGE_PYTHON_REQUIRED TRUE)
    set(FERN_FERN_LANGUAGE_SCRIPT_REQUIRED TRUE)
    set(FERN_FERN_LANGUAGE_UNCERTAINTY_REQUIRED TRUE)
endif()


if(FERN_BUILD_PYTHON)
    # Required third party software.
    set(DEVBASE_NUMPY_REQUIRED TRUE)
    set(DEVBASE_PYTHON_LIBS_REQUIRED TRUE)
    set(DEVBASE_SWIG_REQUIRED TRUE)

    list(APPEND DEVBASE_REQUIRED_BOOST_COMPONENTS python)

    # Required Fern targets.
    set(FERN_FERN_PYTHON_REQUIRED TRUE)
endif()


if(FERN_BUILD_TEST)
    set(DEVBASE_BUILD_TEST TRUE)
    set(DEVBASE_BOOST_REQUIRED TRUE)
    list(APPEND DEVBASE_REQUIRED_BOOST_COMPONENTS
        system unit_test_framework)
endif()


include(CheckIncludeFile)
include(CheckSymbolExists)


# This header is used by the malloc_test tool sources.
check_include_file(malloc.h FERN_MALLOC_H_EXISTS)
if(FERN_MALLOC_H_EXISTS)
   check_symbol_exists(M_TRIM_THRESHOLD malloc.h
       FERN_REQUIRED_MALLOC_SYMBOLS_AVAILABLE)
endif()


# make_unique not available in gcc <= 4.8, but we have our own.
include(CheckCXXSourceRuns)
check_cxx_source_runs("
    #include <memory>

    int main(int, char**)
    {
        auto silly = std::make_unique<int>(5);
    }"
    FERN_COMPILER_HAS_MAKE_UNIQUE
)
if(NOT FERN_COMPILER_HAS_MAKE_UNIQUE)
    set(FERN_COMPILER_DOES_NOT_HAVE_MAKE_UNIQUE TRUE)
endif()


# Link errors on gcc-4.8. Incomplete support for regex.
# If regex is not usable, we cannot use String::split and all code
# that depends on it. That is fine if we are only building for Fern.Algorithm,
# but may not be fine for some other components.
check_cxx_source_runs("
    #include <regex>
    #include <string>
    #include <vector>

    int main(int, char**)
    {
        std::string silly;
        std::regex regular_expression(\"blah\");
        std::vector<std::string> words;

        std::copy_if(std::sregex_token_iterator(silly.begin(), silly.end(),
            regular_expression, -1), std::sregex_token_iterator(),
            std::back_inserter(words),
            [](std::string const& string) { return !string.empty(); });

        // std::string s = \"Some people\";

        // std::regex self_regex(\"REGULAR EXPRESSIONS\",
        //         std::regex_constants::ECMAScript | std::regex_constants::icase);
        // if (std::regex_search(s, self_regex)) {
        //     // std::cout << \"Text contains the phrase\";
        // }
    }"
    FERN_COMPILER_HAS_REGEX
)
if(NOT FERN_COMPILER_HAS_REGEX)
    set(FERN_COMPILER_DOES_NOT_HAVE_REGEX TRUE)
endif()
