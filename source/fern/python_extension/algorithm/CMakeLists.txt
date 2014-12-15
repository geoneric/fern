ADD_SUBDIRECTORY(algebra)
ADD_SUBDIRECTORY(core)
ADD_SUBDIRECTORY(space)
ADD_SUBDIRECTORY(test)


SET(EXTENSION_SOURCES
    execution_policy
    python_extension
)
ADD_LIBRARY(_fern_algorithm SHARED ${EXTENSION_SOURCES})
TARGET_LINK_LIBRARIES(_fern_algorithm
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
