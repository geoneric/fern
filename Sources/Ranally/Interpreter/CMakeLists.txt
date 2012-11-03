INCLUDE(UseBoost)
INCLUDE(UseIcu)
INCLUDE(UseLoki)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_BINARY_DIR}/../..
)

ADD_SUBDIRECTORY(Tests)

SET(SOURCES
    Interpreter.cc
)
ADD_LIBRARY(RanallyInterpreter
    ${SOURCES}
)
ADD_DEPENDENCIES(RanallyInterpreter
    # This header is generated in the Operation project.
    Operation-xml.h
)
