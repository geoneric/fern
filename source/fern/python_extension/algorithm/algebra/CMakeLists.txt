add_subdirectory(elementary)

configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/algebra/__init__.py
    COPYONLY
)
