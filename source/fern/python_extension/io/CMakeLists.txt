ADD_SUBDIRECTORY(test)
# SET(LIB_SOURCES
#     masked_raster
# )
# ADD_LIBRARY(fern_python_extension_io ${LIB_SOURCES})
# TARGET_LINK_LIBRARIES(fern_python_extension_io
#     stdc++
# )
SET(EXTENSION_SOURCES
    gdal
    python_extension
)
ADD_LIBRARY(_fern_io SHARED ${EXTENSION_SOURCES})
TARGET_LINK_LIBRARIES(_fern_io
    fern_python_extension_feature
    fern_feature_core
    # fern_io_core
    ${GDAL_LIBRARIES}
    ${Boost_PYTHON_LIBRARY}
    stdc++
)
CONFIGURE_PYTHON_EXTENSION(_fern_io "_fern_io")
SET_TARGET_PROPERTIES(_fern_io
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/io
)
CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/io/__init__.py
    COPYONLY
)
