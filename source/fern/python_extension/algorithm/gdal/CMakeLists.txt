SET(SOURCES
    add
    add_overloads
    algorithm
    python_extension
    slope
    slope_overloads
    util
    wrapped_data_type
)
ADD_LIBRARY(fern_algorithm_gdal SHARED ${SOURCES})
TARGET_LINK_LIBRARIES(fern_algorithm_gdal
    _fern_algorithm_numpy
    fern_python_extension_algorithm_core
    fern_python_extension_core
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
