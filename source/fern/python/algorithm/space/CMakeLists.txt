add_subdirectory(focal)

configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/space/__init__.py
    COPYONLY
)
add_custom_target(fern_algorithm_space_python_package
    DEPENDS
        ${PROJECT_BINARY_DIR}/bin/fern/algorithm/space/__init__.py
)
add_dependencies(fern_algorithm_space_python_package
    fern_algorithm_space_focal_python_package
)
