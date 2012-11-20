INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_BINARY_DIR}/../..
)
ADD_EXECUTABLE(ranally
    Command
    ConvertCommand
    DescribeCommand
    ExecuteCommand
    ImportCommand
    Ranally
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
