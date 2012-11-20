#include "Ranally/Interpreter/interpreter.h"
#include "Ranally/Operation/operation_xml_parser.h"
#include "Ranally/Operation/operation-xml.h"
#include "Ranally/Language/AnnotateVisitor.h"
#include "Ranally/Language/ExecuteVisitor.h"
#include "Ranally/Language/IdentifyVisitor.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/ThreadVisitor.h"
#include "Ranally/Language/ValidateVisitor.h"


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
