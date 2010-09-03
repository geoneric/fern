INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_BINARY_DIR}
)
ADD_CUSTOM_COMMAND(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/Ranally-pskel.hxx
    ${CMAKE_CURRENT_BINARY_DIR}/Ranally-pskel.cxx
  COMMAND
    ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
      ${CMAKE_CURRENT_SOURCE_DIR}/Ranally.xsd
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Ranally.xsd
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating XML parser ѕkeleton" VERBATIM
)
SET(SOURCES
  ${CMAKE_CURRENT_BINARY_DIR}/Ranally-pskel.cxx
  AlgebraParser.cc
  ExpressionVertex.cc
  NameVertex.cc
  SyntaxVertex.cc
  XmlParser.cc
)
ADD_LIBRARY(RanallyLanguage
  ${SOURCES}
)
SET(TEST_SOURCES
  AlgebraParserTest.cc
  ExpressionVertexTest.cc
  NameVertexTest.cc
  SyntaxVertexTest.cc
  Test.cc
  XmlParserTest.cc
)
ADD_EXECUTABLE(testRanallyLanguage
  ${TEST_SOURCES}
)
TARGET_LINK_LIBRARIES(testRanallyLanguage
  RanallyLanguage
  ${DEVENV_UTILS_LIBRARY}
  ${DEVENV_UNICODE_UTILS_LIBRARY}
  ${PYTHON_LIBRARIES}
  ${BOOST_REGEX_LIBRARY}
  ${XERCES_LIBRARIES}
  ${EXPAT_LIBRARIES}
  ${ICU_LIBRARIES}
)
