# ADD_SUBDIRECTORY(c_array)
# ADD_SUBDIRECTORY(core)
# ADD_SUBDIRECTORY(gdal)
# ADD_SUBDIRECTORY(numpy)
ADD_SUBDIRECTORY(test)


SET(EXTENSION_SOURCES
    add
    greater
    if_number_raster
    if_raster_number
    if_raster_raster
    multiply
    less
    python_extension
    slope
)
ADD_LIBRARY(_fern_algorithm SHARED ${EXTENSION_SOURCES})
TARGET_LINK_LIBRARIES(_fern_algorithm
    fern_python_extension_feature
    fern_feature_core
    fern_algorithm_core
    fern_core
    ${Boost_PYTHON_LIBRARY}
    stdc++
)
CONFIGURE_PYTHON_EXTENSION(_fern_algorithm "_fern_algorithm")
SET_TARGET_PROPERTIES(_fern_algorithm
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/algorithm
)
CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/__init__.py
    COPYONLY
)
