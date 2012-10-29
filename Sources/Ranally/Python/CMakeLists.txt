INCLUDE(UseBoost)
INCLUDE(UseIcu)
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
# SET(TEST_SOURCES
#     Test.cc
# )
# ADD_EXECUTABLE(testRanallyPython
#     ${TEST_SOURCES}
# )
# ADD_CUSTOM_COMMAND(TARGET testRanallyPython POST_BUILD
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMAND testRanallyPython
# )
