ADD_SUBDIRECTORY(focal)

CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/space/__init__.py
    COPYONLY
)
