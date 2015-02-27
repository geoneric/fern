if(FERN_TEST)
    add_subdirectory(test)
endif()


set(EXTENSION_SOURCES
    if_number_raster
    if_raster_number
    if_raster_raster
    python_extension
)
add_library(_fern_algorithm_core SHARED ${EXTENSION_SOURCES})
target_link_libraries(_fern_algorithm_core
    fern_python_extension_feature
    fern_feature_core
    fern_algorithm_policy
    fern_algorithm_core
    fern_core
    ${Boost_PYTHON_LIBRARY}
    stdc++
)
configure_python_extension(_fern_algorithm_core "_fern_algorithm_core")
set_target_properties(_fern_algorithm_core
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/algorithm/core
)
configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/core/__init__.py
    COPYONLY
)
