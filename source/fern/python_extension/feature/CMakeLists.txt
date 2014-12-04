ADD_SUBDIRECTORY(test)
# SET(LIB_SOURCES
# )
# ADD_LIBRARY(fern_python_extension_feature ${LIB_SOURCES})
# TARGET_LINK_LIBRARIES(fern_python_extension_feature
#     stdc++
# )
SET(EXTENSION_SOURCES
    masked_raster
    python_extension
)
ADD_LIBRARY(_fern_feature SHARED ${EXTENSION_SOURCES})
TARGET_LINK_LIBRARIES(_fern_feature
    # fern_python_extension_feature
    fern_feature_core
    ${Boost_PYTHON_LIBRARY}
    stdc++
)
CONFIGURE_PYTHON_EXTENSION(_fern_feature "_fern_feature")
SET_TARGET_PROPERTIES(_fern_feature
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/feature
)
CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/feature/__init__.py
    COPYONLY
)
