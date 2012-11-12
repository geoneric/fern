INCLUDE(UseBoost)
INCLUDE(UseExpat)
INCLUDE(UseIcu)
INCLUDE(UseLoki)
INCLUDE(UsePython)
SET(XSD_ROOT ${PCRTEAM_PLATFORM})
INCLUDE(UseXsd)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_BINARY_DIR}/../..
)

ADD_SUBDIRECTORY(Tests)

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
            --namespace-map http://www.jemig.eu/ranally=ranally
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
    OptimizeVisitor.cc
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
TARGET_LINK_LIBRARIES(RanallyLanguage
    RanallyOperation
    RanallyPython
    RanallyUtil
    ${EXPAT_LIBRARIES}
    ${PYTHON_LIBRARIES}
    ${ICU_LIBRARIES}
)
