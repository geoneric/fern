# Options for selecting the modules to build.
# FERN_BUILD_<module>

# Options for selecting features.
# FERN_WITH_<feature>

option(FERN_BUILD_ALL "Build everything" FALSE)
option(FERN_WITH_ALL "Support all features" FALSE)

option(FERN_BUILD_ALGORITHM "Build Fern.Algorithm module" FALSE)

option(FERN_BUILD_IO "Build Fern.IO module" FALSE)
option(FERN_IO_WITH_GDAL "Add support for GDAL" FALSE)
option(FERN_IO_WITH_GPX "Add support for GPX" FALSE)
option(FERN_IO_WITH_HDF5 "Add support for HDF5" FALSE)
option(FERN_IO_WITH_NETCDF "Add support for NetCDF" FALSE)

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
    set(FERN_BUILD_DOCUMENTATION TRUE)
    set(FERN_BUILD_IO TRUE)
    set(FERN_BUILD_LANGUAGE TRUE)
    set(FERN_BUILD_PYTHON TRUE)
    set(FERN_BUILD_TEST TRUE)
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
    set(FERN_LANGUAGE_WITH_INTERPRETER TRUE)
    set(FERN_LANGUAGE_WITH_COMPILER TRUE)
    set(FERN_IO_WITH_GDAL TRUE)
    set(FERN_IO_WITH_GPX TRUE)
    set(FERN_IO_WITH_HDF5 TRUE)
    set(FERN_IO_WITH_NETCDF TRUE)
endif()

if(FERN_IO_WITH_ALL)
    set(FERN_IO_WITH_GDAL TRUE)
    set(FERN_IO_WITH_GPX TRUE)
    set(FERN_IO_WITH_HDF5 TRUE)
    set(FERN_IO_WITH_NETCDF TRUE)
endif()

if(FERN_IO_WITH_NETCDF)
    set(FERN_IO_WITH_HDF5 TRUE)
endif()


# Fern targets that can be built. Depending on the selection made during
# configuration, some targets will be built and others won't.
set(FERN_FERN_ALGORITHM_REQUIRED FALSE)
set(FERN_FERN_CORE_REQUIRED FALSE)
set(FERN_FERN_EXAMPLE_REQUIRED FALSE)
set(FERN_FERN_EXPRESSION_TREE_REQUIRED FALSE)
set(FERN_FERN_FEATURE_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_AST_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_BACK_END_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_COMMAND_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_COMPILER_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_CORE_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_EXECUTOR_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_INTERPRETER_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_IO_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_FEATURE_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_OPERATION_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_PYTHON_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_SCRIPT_REQUIRED FALSE)
set(FERN_FERN_LANGUAGE_UNCERTAINTY_REQUIRED FALSE)
set(FERN_FERN_IO_REQUIRED FALSE)
set(FERN_FERN_PYTHON_REQUIRED FALSE)
set(FERN_FERN_UNCERTAINTY_REQUIRED FALSE)


# Third party software that is used by some Fern targets. Depending on the
# selection of Fern software made during configuration, some third party
# software is required and other isn't.
set(FERN_BOOST_REQUIRED FALSE)
set(FERN_EXPAT_REQUIRED FALSE)
set(FERN_GDAL_REQUIRED FALSE)
set(FERN_HDF5_REQUIRED FALSE)
set(FERN_LOKI_REQUIRED FALSE)
set(FERN_NETCDF_REQUIRED FALSE)
set(FERN_NUMPY_REQUIRED FALSE)
set(FERN_PYTHON_INTERP_REQUIRED FALSE)
set(FERN_PYTHON_LIBS_REQUIRED FALSE)
set(FERN_READLINE_REQUIRED FALSE)
set(FERN_SWIG_REQUIRED FALSE)
set(FERN_XERCES_REQUIRED FALSE)
set(FERN_XSD_REQUIRED FALSE)


if(FERN_BUILD_ALGORITHM)
    # Required third party software.
    set(FERN_BOOST_REQUIRED TRUE)
    list(APPEND FERN_REQUIRED_BOOST_COMPONENTS
        filesystem system timer)

    # Required Fern targets.
    set(FERN_FERN_ALGORITHM_REQUIRED TRUE)
    set(FERN_FERN_CORE_REQUIRED TRUE)
    set(FERN_FERN_EXAMPLE_REQUIRED TRUE)
    set(FERN_FERN_FEATURE_REQUIRED TRUE)
endif()


# if(FERN_HPX)
#     # Required third party software.
#     set(FERN_BOOST_REQUIRED TRUE)
#     list(APPEND FERN_REQUIRED_BOOST_COMPONENTS
#         date_time program_options regex serialization thread chrono)
#     set(FERN_HPX_REQUIRED TRUE)
# 
#     # Required Fern targets.
#     set(FERN_FERN_HPX_REQUIRED TRUE)
# endif()


if(FERN_BUILD_IO)
    # Required third party software.
    set(FERN_BOOST_REQUIRED TRUE)
    list(APPEND FERN_REQUIRED_BOOST_COMPONENTS
        filesystem system timer)
    if(FERN_IO_WITH_GDAL)
        set(FERN_GDAL_REQUIRED TRUE)
    endif()
    if(FERN_IO_WITH_GPX)
        set(FERN_XERCES_REQUIRED TRUE)
        set(FERN_XSD_REQUIRED TRUE)
    endif()
    if(FERN_IO_WITH_HDF5)
        set(FERN_HDF5_REQUIRED TRUE)
    endif()
    if(FERN_IO_WITH_NETCDF)
        set(FERN_NETCDF_REQUIRED TRUE)
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
    set(FERN_BOOST_REQUIRED TRUE)
    list(APPEND FERN_REQUIRED_BOOST_COMPONENTS
        filesystem system timer)
    set(FERN_EXPAT_REQUIRED TRUE)
    set(FERN_LOKI_REQUIRED TRUE)
    set(FERN_PYTHON_LIBS_REQUIRED TRUE)
    set(FERN_READLINE_REQUIRED TRUE)
    set(FERN_XSD_REQUIRED TRUE)

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
    set(FERN_NUMPY_REQUIRED TRUE)
    set(FERN_PYTHON_LIBS_REQUIRED TRUE)
    set(FERN_SWIG_REQUIRED TRUE)

    list(APPEND FERN_REQUIRED_BOOST_COMPONENTS python)

    # Required Fern targets.
    set(FERN_FERN_PYTHON_REQUIRED TRUE)
endif()


if(FERN_BUILD_TEST)
    set(FERN_BOOST_REQUIRED TRUE)
    list(APPEND FERN_REQUIRED_BOOST_COMPONENTS
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
