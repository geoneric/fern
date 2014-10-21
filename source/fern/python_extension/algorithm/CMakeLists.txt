ADD_SUBDIRECTORY(c_array)
ADD_SUBDIRECTORY(gdal)
ADD_SUBDIRECTORY(numpy)


CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/__init__.py
    COPYONLY
)
