ADD_SUBDIRECTORY(test)


SET(EXTENSION_SOURCES
    if_number_raster
    if_raster_number
    if_raster_raster
    python_extension
)
ADD_LIBRARY(_fern_algorithm_core SHARED ${EXTENSION_SOURCES})
TARGET_LINK_LIBRARIES(_fern_algorithm_core
    fern_python_extension_feature
    fern_feature_core
    fern_algorithm_core
    fern_core
    ${Boost_PYTHON_LIBRARY}
    stdc++
)
CONFIGURE_PYTHON_EXTENSION(_fern_algorithm_core "_fern_algorithm_core")
SET_TARGET_PROPERTIES(_fern_algorithm_core
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/algorithm/core
)
CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/core/__init__.py
    COPYONLY
)
