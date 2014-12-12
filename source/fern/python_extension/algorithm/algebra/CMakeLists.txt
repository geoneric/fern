ADD_SUBDIRECTORY(elementary)

CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/algebra/__init__.py
    COPYONLY
)
