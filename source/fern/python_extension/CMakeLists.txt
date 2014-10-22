ADD_SUBDIRECTORY(algorithm)
ADD_SUBDIRECTORY(core)


CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/__init__.py
    COPYONLY
)
