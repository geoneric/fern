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
  # During the build of this target, source files are generated. At least one
  # of these files is included by the interpreter's source (Operation-xml.h).
  RanallyOperation
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
