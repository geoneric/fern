ADD_EXECUTABLE(fern
    command
    compile_command
    describe_command
    execute_command
    import_command
    interpreter
    message
    geoneric
)
ADD_DEPENDENCIES(fern
    Operation-xml.h
)
TARGET_LINK_LIBRARIES(fern
    geoneric_compiler
    geoneric_interpreter
    ${READLINE_LIBRARY}
    termcap
)
INSTALL(
    TARGETS fern
    RUNTIME DESTINATION bin COMPONENT Runtime
)
