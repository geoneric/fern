ADD_EXECUTABLE(fern
    command
    compile_command
    describe_command
    execute_command
    import_command
    interpreter
    message
    fern
)
ADD_DEPENDENCIES(fern
    Operation-xml.h
)
TARGET_LINK_LIBRARIES(fern
    fernlib
    ${READLINE_LIBRARY}
    termcap
)
INSTALL(
    TARGETS fern
    EXPORT fern_targets
    RUNTIME DESTINATION bin
)
SET(APPLICATIONS
    "\${CMAKE_INSTALL_PREFIX}/bin/fern"
)
