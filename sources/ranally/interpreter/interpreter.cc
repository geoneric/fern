#include "ranally/interpreter/interpreter.h"
#include <sstream>
#include "ranally/core/exception.h"
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

    : _algebra_parser(),
      _xml_parser()

{
}


//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception ParseError
  \warning   .
  \sa        .
*/
ScriptVertexPtr Interpreter::parse_string(
    String const& string) const
{
    ScriptVertexPtr script_vertex;
    try {
        script_vertex = _xml_parser.parse(_algebra_parser.parse_string(string));
    }
    catch(detail::ParseError const& exception) {
        String const* exception_message = boost::get_error_info<
            detail::ExceptionMessage>(exception);
        assert(exception_message);
        throw ParseError(*exception_message);
    }

    return script_vertex;
}


//! Read script from \a filename.
/*!
  \param     filename Name of file to read script from.
  \exception ParseError

  In case \a filename is empty, the script is read from standard input.
*/
ScriptVertexPtr Interpreter::parse_file(
    String const& filename) const
{
    ScriptVertexPtr script_vertex;

    if(filename.is_empty()) {
        // Read script from the standard input stream.
        // Exception are handled by parse_string(...).
        std::ostringstream stream;
        stream << std::cin.rdbuf();
        script_vertex = parse_string(stream.str());
    }
    else {
        try {
            // Read script from a file.
            script_vertex = _xml_parser.parse(_algebra_parser.parse_file(
                filename));
        }
        catch(detail::ParseError const& exception) {
            String const* exception_filename = boost::get_error_info<
                detail::ExceptionFilename>(exception);
            assert(exception_filename);
            String const* exception_message = boost::get_error_info<
                detail::ExceptionMessage>(exception);
            assert(exception_message);
            throw ParseError(*exception_filename, *exception_message);
        }
    }

    return script_vertex;
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
    ScriptVertexPtr const& tree) const
{
    ThreadVisitor thread_visitor;
    tree->Accept(thread_visitor);

    IdentifyVisitor identify_visitor;
    tree->Accept(identify_visitor);

    OperationsPtr operations(OperationXmlParser().parse(operations_xml));
    AnnotateVisitor annotate_visitor(operations);
    tree->Accept(annotate_visitor);
}


//! Validate the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to validate.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Annotation (see annotate(ScriptVertexPtr const&).
  - Validation.
*/
void Interpreter::validate(
    ScriptVertexPtr const& tree) const
{
    annotate(tree);
    ValidateVisitor validate_visitor;
    tree->Accept(validate_visitor);
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
    ScriptVertexPtr const& tree) const
{
    validate(tree);
    ExecuteVisitor execute_visitor;
    tree->Accept(execute_visitor);
}

} // namespace ranally
