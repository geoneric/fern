INCLUDE(UseBoost)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_BINARY_DIR}/../..
)
ADD_EXECUTABLE(ranally
    Command.cc
    ConvertCommand.cc
    DescribeCommand.cc
    ExecuteCommand.cc
    ImportCommand.cc
    Ranally.cc
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
