#include "ranally/interpreter/interpreter.h"
#include "ranally/operation/operation_xml_parser.h"
#include "ranally/operation/operation-xml.h"
#include "ranally/language/annotate_visitor.h"
#include "ranally/language/execute_visitor.h"
#include "ranally/language/identify_visitor.h"
#include "ranally/language/script_vertex.h"
#include "ranally/language/thread_visitor.h"
#include "ranally/language/validate_visitor.h"


namespace ranally {

Interpreter::Interpreter()

    : _algebraParser(),
      _xmlParser()

{
}


ScriptVertexPtr Interpreter::parseString(
    String const& string)
{
    return _xmlParser.parse(_algebraParser.parseString(string));
}


//! Annotate the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to annotate.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Threading.
  - Identification.
  - Annotation.
*/
void Interpreter::annotate(
    ScriptVertexPtr const& tree)
{
    ThreadVisitor threadVisitor;
    tree->Accept(threadVisitor);

    IdentifyVisitor identifyVisitor;
    tree->Accept(identifyVisitor);

    OperationsPtr operations(OperationXmlParser().parse(operationsXml));
    AnnotateVisitor annotateVisitor(operations);
    tree->Accept(annotateVisitor);
}


//! Validate the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to validate.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Annotation (seee annotate(ScriptVertexPtr const&).
  - Validation.
*/
void Interpreter::validate(
    ScriptVertexPtr const& tree)
{
    annotate(tree);
    ValidateVisitor validateVisitor;
    tree->Accept(validateVisitor);
}


//! Execute the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to execute.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Validation (see validate(ScriptVertexPtr const&).
  - Execution.
*/
void Interpreter::execute(
    ScriptVertexPtr const& tree)
{
    validate(tree);
    ExecuteVisitor executeVisitor;
    tree->Accept(executeVisitor);
}

} // namespace ranally
