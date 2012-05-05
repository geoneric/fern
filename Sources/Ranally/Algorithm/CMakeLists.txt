INCLUDE(UseBoost)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
SET(TEST_SOURCES
  AverageTest.cc
  PlusTest.cc
  Test.cc
)
ADD_EXECUTABLE(testRanallyAlgorithm
  ${TEST_SOURCES}
)
# ADD_CUSTOM_COMMAND(TARGET testRanallyAlgorithm POST_BUILD
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#   COMMAND testRanallyAlgorithm
# )
ADD_CUSTOM_TARGET(runTestRanallyAlgorithm
  testRanallyAlgorithm
  DEPENDS testRanallyAlgorithm
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running RanallyAlgorithm unit tests" VERBATIM
)
ADD_DEPENDENCIES(tests runTestRanallyAlgorithm)

