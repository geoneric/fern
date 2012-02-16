INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
SET(SOURCES
  Attribute.cc
  Domain.cc
  Feature.cc
  Value.cc
)
ADD_LIBRARY(RanallyIO
  ${SOURCES}
)
# SET(TEST_SOURCES
#   Test.cc
# )
# ADD_EXECUTABLE(testRanallyIO
#   ${TEST_SOURCES}
# )
# TARGET_LINK_LIBRARIES(testRanallyIO
#   RanallyIO
# )
