#include "Ranally/Interpreter/Interpreter.h"
#include "Ranally/Operation/XmlParser.h"
#include "Ranally/Operation/Operation-xml.h"
#include "Ranally/Language/AnnotateVisitor.h"
#include "Ranally/Language/ExecuteVisitor.h"
#include "Ranally/Language/IdentifyVisitor.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/ThreadVisitor.h"
#include "Ranally/Language/ValidateVisitor.h"



namespace ranally {
namespace interpreter {

Interpreter::Interpreter()

  : _algebraParser(),
    _xmlParser()

{
}



Interpreter::~Interpreter()
{
}



language::ScriptVertexPtr Interpreter::parseString(
  UnicodeString const& string)
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
  language::ScriptVertexPtr const& tree)
{
  language::ThreadVisitor threadVisitor;
  tree->Accept(threadVisitor);

  language::IdentifyVisitor identifyVisitor;
  tree->Accept(identifyVisitor);

  operation::OperationsPtr operations(operation::XmlParser().parse(
    operation::operationsXml));
  language::AnnotateVisitor annotateVisitor(operations);
  tree->Accept(annotateVisitor);
}



//! Validate the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to validate.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Annotation (seee annotate(language::ScriptVertexPtr const&).
  - Validation.
*/
void Interpreter::validate(
  language::ScriptVertexPtr const& tree)
{
  annotate(tree);
  language::ValidateVisitor validateVisitor;
  tree->Accept(validateVisitor);
}



//! Execute the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to execute.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Validation (see validate(language::ScriptVertexPtr const&).
  - Execution.
*/
void Interpreter::execute(
  language::ScriptVertexPtr const& tree)
{
  validate(tree);

  language::ExecuteVisitor executeVisitor;
  tree->Accept(executeVisitor);
}

} // namespace interpreter
} // namespace ranally

