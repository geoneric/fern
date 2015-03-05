add_subdirectory(elementary)

configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/algebra/__init__.py
    COPYONLY
)
add_custom_target(fern_algorithm_algebra_python_package
    DEPENDS
        ${PROJECT_BINARY_DIR}/bin/fern/algorithm/algebra/__init__.py
)
add_dependencies(fern_algorithm_algebra_python_package
    fern_algorithm_algebra_elementary_python_package
)
