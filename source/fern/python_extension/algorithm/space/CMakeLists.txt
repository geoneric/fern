add_subdirectory(focal)

configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/space/__init__.py
    COPYONLY
)
