SET(TEST_SOURCES
  ExampleCodeTest.cc
  Test.cc
)
ADD_EXECUTABLE(testExampleCode
  ${TEST_SOURCES}
)
TARGET_LINK_LIBRARIES(testExampleCode
  ${BOOST_UNIT_TEST_FRAMEWORK_LIBRARY}
)
CONFIGURE_TEST(testExampleCode)
ADD_TEST(testExampleCode ${EXECUTABLE_OUTPUT_PATH}/testExampleCode)
