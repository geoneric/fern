# SET(_LINK_LIBRARIES
#   RanallyUtil
#   ${EXPAT_LIBRARIES}
#   ${ICU_LIBRARIES}
# )
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_BINARY_DIR}/../..
)
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
SET(TEST_SOURCES
  InterpreterTest.cc
  Test.cc
)
ADD_EXECUTABLE(testRanallyInterpreter
  ${TEST_SOURCES}
)
TARGET_LINK_LIBRARIES(testRanallyInterpreter
  RanallyLanguage
  RanallyOperation
  # ${_LINK_LIBRARIES}
)
ADD_CUSTOM_TARGET(runTestRanallyInterpreter
  testRanallyInterpreter
  DEPENDS testRanallyInterpreter
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running RanallyInterpreter unit tests" VERBATIM
)
