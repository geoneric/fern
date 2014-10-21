SET(SOURCES
    add
    add_overloads
    error
    python_extension
    swig
    util
    wrapped_data_type
)
ADD_LIBRARY(fern_algorithm_gdal SHARED ${SOURCES})
ADD_DEPENDENCIES(fern_algorithm_gdal swig_runtime)
TARGET_LINK_LIBRARIES(fern_algorithm_gdal
    fern_algorithm_core
    fern_feature_core
    ${GDAL_LIBRARY}
    stdc++
)
CONFIGURE_PYTHON_EXTENSION(fern_algorithm_gdal "_fern_algorithm_gdal")
SET_TARGET_PROPERTIES(fern_algorithm_gdal
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/algorithm/gdal
)
CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/gdal/__init__.py
    COPYONLY
)
