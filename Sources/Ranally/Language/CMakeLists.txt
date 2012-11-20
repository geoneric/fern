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
    AlgebraParser
    AnnotateVisitor
    AssignmentVertex
    AstDotVisitor
    CopyVisitor
    DotVisitor
    ExecuteVisitor
    ExpressionVertex
    FunctionVertex
    IdentifyVisitor
    IfVertex
    FlowgraphDotVisitor
    NameVertex
    NumberVertex
    OperationVertex
    OperatorVertex
    OptimizeVisitor
    ScriptVertex
    ScriptVisitor
    StatementVertex
    StringVertex
    SymbolTable
    SyntaxVertex
    ThreadVisitor
    ValidateVisitor
    Visitor
    WhileVertex
    XmlParser
)
ADD_LIBRARY(RanallyLanguage
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(RanallyLanguage
    RanallyOperation
    RanallyPython
    RanallyUtil
    ${EXPAT_LIBRARY}
    ${PYTHON_LIBRARIES}
)
