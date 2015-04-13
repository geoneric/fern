add_test_conditionally(test)

set(LIB_SOURCES
    masked_raster
)
add_library(fern_python_extension_feature ${LIB_SOURCES})
target_link_libraries(fern_python_extension_feature
    stdc++
)
set(EXTENSION_SOURCES
    numpy
    python_extension
)
add_library(_fern_feature SHARED ${EXTENSION_SOURCES})
target_link_libraries(_fern_feature
    fern_python_extension_feature
    fern_feature_core
    ${Boost_PYTHON_LIBRARY}
    ${PYTHON_LIBRARIES}
    stdc++
)
configure_python_extension(_fern_feature "_fern_feature")
set_target_properties(_fern_feature
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/feature
)
configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/feature/__init__.py
    COPYONLY
)
add_custom_target(fern_feature_python_package
    DEPENDS
        ${PROJECT_BINARY_DIR}/bin/fern/feature/__init__.py
)
add_dependencies(fern_feature_python_package
    _fern_feature
)
