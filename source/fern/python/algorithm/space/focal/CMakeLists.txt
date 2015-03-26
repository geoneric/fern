if(FERN_TEST)
    add_subdirectory(test)
endif()


set(EXTENSION_SOURCES
    python_extension
    slope
)
add_library(_fern_algorithm_space_focal SHARED ${EXTENSION_SOURCES})
target_link_libraries(_fern_algorithm_space_focal
    fern_python_extension_feature
    fern_feature_core
    fern_algorithm_policy
    fern_algorithm_core
    fern_core
    ${Boost_PYTHON_LIBRARY}
    ${PYTHON_LIBRARIES}
    stdc++
)
configure_python_extension(_fern_algorithm_space_focal
    "_fern_algorithm_space_focal")
set_target_properties(_fern_algorithm_space_focal
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/bin/fern/algorithm/space/focal
)
configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/space/focal/__init__.py
    COPYONLY
)
add_custom_target(fern_algorithm_space_focal_python_package
    DEPENDS
        ${PROJECT_BINARY_DIR}/bin/fern/algorithm/space/focal/__init__.py
)
add_dependencies(fern_algorithm_space_focal_python_package
    _fern_algorithm_space_focal
)
