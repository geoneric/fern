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
# TARGET_LINK_LIBRARIES(testRanallyLanguage
#   RanallyLanguage
# )
