add_subdirectory(algorithm)
add_subdirectory(core)
add_subdirectory(feature)
add_subdirectory(io)
add_subdirectory(test)


configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/__init__.py
    COPYONLY
)
