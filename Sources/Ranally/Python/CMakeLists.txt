INCLUDE(UseBoost)
INCLUDE(UsePython)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
SET(SOURCES
    BorrowedReference.cc
    Client.cc
    Exception.cc
    OwnedReference.cc
    String.cc
)
ADD_LIBRARY(RanallyPython
    ${SOURCES}
)
