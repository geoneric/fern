ADD_SUBDIRECTORY(test)


SET(EXTENSION_SOURCES
    python_extension
    slope
)
ADD_LIBRARY(_fern_algorithm_space_focal SHARED ${EXTENSION_SOURCES})
TARGET_LINK_LIBRARIES(_fern_algorithm_space_focal
    fern_python_extension_feature
    fern_feature_core
    fern_algorithm_core
    fern_core
    ${Boost_PYTHON_LIBRARY}
    stdc++
)
CONFIGURE_PYTHON_EXTENSION(_fern_algorithm_space_focal
    "_fern_algorithm_space_focal")
SET_TARGET_PROPERTIES(_fern_algorithm_space_focal
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/bin/fern/algorithm/space/focal
)
CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/space/focal/__init__.py
    COPYONLY
)
