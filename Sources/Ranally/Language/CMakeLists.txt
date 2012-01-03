SET(_LINK_LIBRARIES
  ${DEVENV_UTILS_LIBRARY}
  ${DEVENV_UNICODE_UTILS_LIBRARY}
  RanallyOperation
  RanallyPython
  ${PYTHON_LIBRARIES}
  ${BOOST_FILESYSTEM_LIBRARY}
  ${BOOST_PROGRAM_OPTIONS_LIBRARY}
  ${BOOST_REGEX_LIBRARY}
  ${BOOST_SYSTEM_LIBRARY}
  ${EXPAT_LIBRARIES}
  ${ICU_LIBRARIES}
)
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_BINARY_DIR}
  # Configure.h
  ${CMAKE_CURRENT_BINARY_DIR}/../..
)
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/SyntaxTree.map.in
  ${CMAKE_CURRENT_BINARY_DIR}/SyntaxTree.map
)
ADD_CUSTOM_COMMAND(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/Ranally-pskel.hxx
    ${CMAKE_CURRENT_BINARY_DIR}/Ranally-pskel.cxx
  COMMAND
    ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
      --type-map ${CMAKE_CURRENT_BINARY_DIR}/SyntaxTree.map
      --namespace-map http://www.jemig.eu/ranally=ranally::language
      ${CMAKE_CURRENT_SOURCE_DIR}/Ranally.xsd
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/SyntaxTree.map
    ${CMAKE_CURRENT_SOURCE_DIR}/Ranally.xsd
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating XML parser ѕkeleton for scripts" VERBATIM
)
SET(SOURCES
  ${CMAKE_CURRENT_BINARY_DIR}/Ranally-pskel.cxx
  AlgebraParser.cc
  AnnotateVisitor.cc
  AssignmentVertex.cc
  AstDotVisitor.cc
  CopyVisitor.cc
  DotVisitor.cc
  ExecuteVisitor.cc
  ExpressionVertex.cc
  FunctionVertex.cc
  IdentifyVisitor.cc
  IfVertex.cc
  FlowgraphDotVisitor.cc
  NameVertex.cc
  NumberVertex.cc
  OperationVertex.cc
  OperatorVertex.cc
  PurifyVisitor.cc
  ScriptVertex.cc
  ScriptVisitor.cc
  StatementVertex.cc
  StringVertex.cc
  SymbolTable.cc
  SyntaxVertex.cc
  ThreadVisitor.cc
  ValidateVisitor.cc
  Visitor.cc
  WhileVertex.cc
  XmlParser.cc
)
ADD_LIBRARY(RanallyLanguage
  ${SOURCES}
)
SET(TEST_SOURCES
  AlgebraParserTest.cc
  AnnotateVisitorTest.cc
  AssignmentVertexTest.cc
  DotVisitorTest.cc
  ExecuteVisitorTest.cc
  ExpressionVertexTest.cc
  FunctionVertexTest.cc
  IdentifyVisitorTest.cc
  IfVertexTest.cc
  NameVertexTest.cc
  NumberVertexTest.cc
  OperatorVertexTest.cc
  PurifyVisitorTest.cc
  ScriptVertexTest.cc
  ScriptVisitorTest.cc
  StatementVertexTest.cc
  StringVertexTest.cc
  SymbolTableTest.cc
  SyntaxVertexTest.cc
  Test.cc
  ThreadVisitorTest.cc
  ValidateVisitorTest.cc
  VisitorTest.cc
  WhileVertexTest.cc
  XmlParserTest.cc
)
ADD_EXECUTABLE(testRanallyLanguage
  ${TEST_SOURCES}
)
TARGET_LINK_LIBRARIES(testRanallyLanguage
  RanallyLanguage
  ${_LINK_LIBRARIES}
)
ADD_DEPENDENCIES(testRanallyLanguage
  Operation-xml.h
)
# ADD_CUSTOM_COMMAND(TARGET testRanallyLanguage POST_BUILD
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#   COMMAND testRanallyLanguage
# )

