SET(LIB_SOURCES
    add
    sqrt
    util
)
ADD_LIBRARY(fern_algorithm_numpy ${LIB_SOURCES})
TARGET_LINK_LIBRARIES(fern_algorithm_numpy
    fern_python_extension_core
    fern_algorithm_core
    fern_feature_core
    stdc++
)
SET(EXTENSION_SOURCES
    add_overloads
    algorithm
    python_extension
    sqrt_overloads
    wrapped_data_type
)
ADD_LIBRARY(_fern_algorithm_numpy SHARED ${EXTENSION_SOURCES})
TARGET_LINK_LIBRARIES(_fern_algorithm_numpy
    fern_algorithm_numpy
    fern_python_extension_algorithm_core
    fern_python_extension_core
    stdc++
)
CONFIGURE_PYTHON_EXTENSION(_fern_algorithm_numpy "_fern_algorithm_numpy")
SET_TARGET_PROPERTIES(_fern_algorithm_numpy
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/algorithm/numpy
)
CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/numpy/__init__.py
    COPYONLY
)
