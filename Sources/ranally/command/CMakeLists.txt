ADD_EXECUTABLE(ranally
    command
    convert_command
    describe_command
    execute_command
    import_command
    ranally
)
ADD_DEPENDENCIES(ranally
    Operation-xml.h
)
TARGET_LINK_LIBRARIES(ranally
    RanallyInterpreter
    RanallyIO
    RanallyLanguage
    RanallyPython
    RanallyOperation
    RanallyUtil
)
INSTALL(
    TARGETS ranally
    RUNTIME DESTINATION bin COMPONENT Runtime
)
