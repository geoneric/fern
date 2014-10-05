OPTION(FERN_ALGORITHM FALSE)
### OPTION(FERN_ALGORITHM_PYTHON_EXTENSION FALSE)
OPTION(FERN_ALL FALSE)


# Third party software that is used by some Fern targets. Depending on the
# selection of Fern software made during configuration, some third party
# software is required and other isn't.
SET(FERN_BOOST_REQUIRED FALSE)
### SET(FERN_CYTHON_REQUIRED FALSE)
SET(FERN_EXPAT_REQUIRED FALSE)
SET(FERN_XSD_REQUIRED FALSE)
SET(FERN_GDAL_REQUIRED FALSE)
SET(FERN_HDF5_REQUIRED FALSE)
SET(FERN_NETCDF_REQUIRED FALSE)
### SET(FERN_NUMPY_REQUIRED FALSE)
SET(FERN_PYTHON_LIBS_REQUIRED FALSE)
SET(FERN_READLINE_REQUIRED FALSE)
SET(FERN_LOKI_REQUIRED FALSE)


IF(FERN_ALL)
    SET(FERN_ALGORITHM TRUE)
    ### SET(FERN_ALGORITHM_PYTHON_EXTENSION TRUE)
ENDIF()


# Fern targets that can be built. Depending on the selection made during
# configuration, some targets will be built and others won't.
SET(FERN_FERN_ALGORITHM_REQUIRED FALSE)
### SET(FERN_FERN_ALGORITHM_PYTHON_EXTENSION_REQUIRED FALSE)
SET(FERN_FERN_AST_REQUIRED FALSE)
SET(FERN_FERN_BACK_END_REQUIRED FALSE)
SET(FERN_FERN_COMMAND_REQUIRED FALSE)
SET(FERN_FERN_COMPILER_REQUIRED FALSE)
SET(FERN_FERN_CORE_REQUIRED FALSE)
SET(FERN_FERN_EXAMPLE_REQUIRED FALSE)
SET(FERN_FERN_EXPRESSION_TREE_REQUIRED FALSE)
SET(FERN_FERN_FEATURE_REQUIRED FALSE)
SET(FERN_FERN_INTERPRETER_REQUIRED FALSE)
SET(FERN_FERN_IO_REQUIRED FALSE)
SET(FERN_FERN_OPERATION_REQUIRED FALSE)
SET(FERN_FERN_PYTHON_REQUIRED FALSE)
SET(FERN_FERN_SCRIPT_REQUIRED FALSE)
SET(FERN_FERN_UNCERTAINTY_REQUIRED FALSE)


### IF(FERN_ALGORITHM OR FERN_ALGORITHM_PYTHON_EXTENSION)
###     # Required third party software.
###     SET(FERN_BOOST_REQUIRED TRUE)
###     LIST(APPEND FERN_REQUIRED_BOOST_COMPONENTS
###         filesystem system timer unit_test_framework)
### 
###     # Required Fern targets.
###     SET(FERN_FERN_ALGORITHM_REQUIRED TRUE)
###     SET(FERN_FERN_CORE_REQUIRED TRUE)
###     SET(FERN_FERN_EXAMPLE_REQUIRED TRUE)
###     SET(FERN_FERN_FEATURE_REQUIRED TRUE)
### ENDIF()


### IF(FERN_ALGORITHM_PYTHON_EXTENSION)
###     # Required third party software.
###     SET(FERN_CYTHON_REQUIRED TRUE)
###     SET(FERN_NUMPY_REQUIRED TRUE)
###     SET(FERN_PYTHON_LIBS_REQUIRED TRUE)
### 
###     # Required Fern targets.
###     SET(FERN_FERN_ALGORITHM_PYTHON_EXTENSION_REQUIRED TRUE)
### ENDIF()


### ELSE()
###     # Build everything.
###     # Required third party software.
###     SET(FERN_BOOST_REQUIRED TRUE)
###     LIST(APPEND FERN_REQUIRED_BOOST_COMPONENTS
###         filesystem system timer unit_test_framework)
###     SET(FERN_EXPAT_REQUIRED TRUE)
###     SET(FERN_XSD_REQUIRED TRUE)
###     SET(FERN_GDAL_REQUIRED TRUE)
###     SET(FERN_HDF5_REQUIRED TRUE)
###     SET(FERN_NETCDF_REQUIRED TRUE)
###     SET(FERN_PYTHON_LIBS_REQUIRED TRUE)
###     SET(FERN_READLINE_REQUIRED TRUE)
###     SET(FERN_LOKI_REQUIRED TRUE)
### 
###     # Required Fern targets.
###     SET(FERN_FERN_ALGORITHM_REQUIRED TRUE)
###     SET(FERN_FERN_AST_REQUIRED TRUE)
###     SET(FERN_FERN_BACK_END_REQUIRED TRUE)
###     SET(FERN_FERN_COMMAND_REQUIRED TRUE)
###     SET(FERN_FERN_COMPILER_REQUIRED TRUE)
###     SET(FERN_FERN_CORE_REQUIRED TRUE)
###     SET(FERN_FERN_EXAMPLE_REQUIRED TRUE)
###     SET(FERN_FERN_EXPRESSION_TREE_REQUIRED TRUE)
###     SET(FERN_FERN_FEATURE_REQUIRED TRUE)
###     SET(FERN_FERN_INTERPRETER_REQUIRED TRUE)
###     SET(FERN_FERN_IO_REQUIRED TRUE)
###     SET(FERN_FERN_OPERATION_REQUIRED TRUE)
###     SET(FERN_FERN_PYTHON_REQUIRED TRUE)
###     SET(FERN_FERN_SCRIPT_REQUIRED TRUE)
###     SET(FERN_FERN_UNCERTAINTY_REQUIRED TRUE)
### ENDIF()


INCLUDE(CheckIncludeFile)
INCLUDE(CheckSymbolExists)


# This header is used by the malloc_test tool sources.
CHECK_INCLUDE_FILE(malloc.h FERN_MALLOC_H_EXISTS)
IF(FERN_MALLOC_H_EXISTS)
   CHECK_SYMBOL_EXISTS(M_TRIM_THRESHOLD malloc.h
       FERN_REQUIRED_MALLOC_SYMBOLS_AVAILABLE)
ENDIF()
