ADD_EXECUTABLE(ranally
    command
    convert_command
    describe_command
    execute_command
    import_command
    interpreter
    message
    ranally
)
ADD_DEPENDENCIES(ranally
    Operation-xml.h
)
TARGET_LINK_LIBRARIES(ranally
    ranally_interpreter
    # ranally_io
    ranally_language
    ranally_python
    ranally_operation
    ranally_core
    ${READLINE_LIBRARY}
    termcap
)
INSTALL(
    TARGETS ranally
    RUNTIME DESTINATION bin COMPONENT Runtime
)
