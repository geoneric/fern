ADD_EXECUTABLE(geoneric
    command
    compile_command
    describe_command
    execute_command
    import_command
    interpreter
    message
    geoneric
)
ADD_DEPENDENCIES(geoneric
    Operation-xml.h
)
TARGET_LINK_LIBRARIES(geoneric
    geoneric_compiler
    geoneric_interpreter
    ${READLINE_LIBRARY}
    termcap
)
INSTALL(
    TARGETS geoneric
    RUNTIME DESTINATION bin COMPONENT Runtime
)
