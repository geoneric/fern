SET(_LINK_LIBRARIES
  # ${DEVENV_UTILS_LIBRARY}
  ${DEVENV_UNICODE_UTILS_LIBRARY}
  # ${PYTHON_LIBRARIES}
  ${BOOST_REGEX_LIBRARY}
  ${EXPAT_LIBRARIES}
  ${ICU_LIBRARIES}
)
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_BINARY_DIR}
  # # Configure.h
  # ${CMAKE_CURRENT_BINARY_DIR}/../..
)
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/Operation.map.in
  ${CMAKE_CURRENT_BINARY_DIR}/Operation.map
)
ADD_CUSTOM_COMMAND(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/Operation-pskel.hxx
    ${CMAKE_CURRENT_BINARY_DIR}/Operation-pskel.cxx
  COMMAND
    ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
      --type-map ${CMAKE_CURRENT_BINARY_DIR}/Operation.map
      --namespace-map http://www.jemig.eu/ranally=ranally::operation
      ${CMAKE_CURRENT_SOURCE_DIR}/Operation.xsd
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/Operation.map
    ${CMAKE_CURRENT_SOURCE_DIR}/Operation.xsd
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating XML parser ѕkeleton for operations" VERBATIM
)
SET(SOURCES
  ${CMAKE_CURRENT_BINARY_DIR}/Operation-pskel.cxx
  Operation.cc
  Operations.cc
  Parameter.cc
  Result.cc
  XmlParser.cc
)
ADD_LIBRARY(RanallyOperation
  ${SOURCES}
)
SET(TEST_SOURCES
  PrintTest.cc
  Test.cc
  XmlParserTest.cc
)
ADD_EXECUTABLE(testRanallyOperation
  ${TEST_SOURCES}
)
TARGET_LINK_LIBRARIES(testRanallyOperation
  RanallyOperation
  ${_LINK_LIBRARIES}
)

