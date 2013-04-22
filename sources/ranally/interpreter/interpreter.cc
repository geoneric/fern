#include "ranally/interpreter/interpreter.h"
#include <sstream>
#include "ranally/core/io_error.h"
#include "ranally/core/parse_error.h"
#include "ranally/core/validate_error.h"
#include "ranally/operation/std/operations.h"
#include "ranally/ast/core/script_vertex.h"
#include "ranally/ast/visitor/identify_visitor.h"
#include "ranally/ast/visitor/thread_visitor.h"


namespace ranally {

Interpreter::Interpreter()

    : _operations(operations()),
      _algebra_parser(),
      _xml_parser(),
      _annotate_visitor(_operations),
      _validate_visitor(),
      _execute_visitor(_operations)

{
}


//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception ParseError In case \a string cannot be parsed.
  \warning   .
  \sa        .
*/
ScriptVertexPtr Interpreter::parse_string(
    String const& string) const
{
    ScriptVertexPtr script_vertex;

    try {
        script_vertex = _xml_parser.parse_string(_algebra_parser.parse_string(
            string));
    }
    catch(detail::ParseError const& exception) {
        String const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        String const* statement = boost::get_error_info<
            detail::ExceptionStatement>(exception);

        String const* message = boost::get_error_info<
            detail::ExceptionMessage>(exception);
        assert(message);

        if(!statement) {
            throw ParseError(*source_name, *line_nr, *col_nr, *message);
        }
        else {
            throw ParseError(*source_name, *line_nr, *col_nr, *statement,
                *message);
        }
    }
    catch(detail::UnsupportedLanguageConstruct const& exception) {
        String const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        String const* construct = boost::get_error_info<
            detail::ExceptionConstruct>(exception);
        assert(construct);

        throw ParseError(*source_name, *line_nr, *col_nr,
            Exception::messages().format_message(
                MessageId::UNSUPPORTED_LANGUAGE_CONSTRUCT, *construct));
    }

    return script_vertex;
}


//! Read script from \a filename.
/*!
  \param     filename Name of file to read script from.
  \exception IOError In case \a filename cannot be opened.
  \exception ParseError In case \a filename cannot be parsed.

  In case \a filename is empty, the script is read from standard input.
*/
ScriptVertexPtr Interpreter::parse_file(
    String const& filename) const
{
    ScriptVertexPtr script_vertex;

    try {
        if(filename.is_empty()) {
            // Read script from the standard input stream.
            // Exceptions are handled by parse_string(...).
            std::ostringstream stream;
            stream << std::cin.rdbuf();
            script_vertex = parse_string(stream.str());
        }
        else {
            // Read script from a file.
            script_vertex = _xml_parser.parse_string(_algebra_parser.parse_file(
                filename));
        }
    }
    catch(detail::IOError const& exception) {
        String const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        int const* errno_ = boost::get_error_info<boost::errinfo_errno>(
            exception);
        assert(errno_);

        throw IOError(*source_name, *errno_);
    }
    catch(detail::ParseError const& exception) {
        String const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        String const* statement = boost::get_error_info<
            detail::ExceptionStatement>(exception);

        String const* message = boost::get_error_info<
            detail::ExceptionMessage>(exception);
        assert(message);

        if(!statement) {
            throw ParseError(*source_name, *line_nr, *col_nr, *message);
        }
        else {
            throw ParseError(*source_name, *line_nr, *col_nr, *statement,
                *message);
        }
    }
    catch(detail::UnsupportedLanguageConstruct const& exception) {
        String const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        String const* construct = boost::get_error_info<
            detail::ExceptionConstruct>(exception);
        assert(construct);

        throw ParseError(*source_name, *line_nr, *col_nr,
            Exception::messages().format_message(
                MessageId::UNSUPPORTED_LANGUAGE_CONSTRUCT, *construct));
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
    ScriptVertexPtr const& tree)
{
    ThreadVisitor thread_visitor;
    tree->Accept(thread_visitor);

    IdentifyVisitor identify_visitor;
    tree->Accept(identify_visitor);

    tree->Accept(_annotate_visitor);
}


//! Validate the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to validate.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Annotation (see annotate(ScriptVertexPtr const&)).
  - Validation.
*/
void Interpreter::validate(
    ScriptVertexPtr const& tree)
{
    try {
        annotate(tree);
        tree->Accept(_validate_visitor);
    }
    catch(detail::UndefinedIdentifier const& exception) {
        String const& source_name = tree->source_name();

        String const* identifier_name = boost::get_error_info<
            detail::ExceptionIdentifier>(exception);
        assert(identifier_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        throw ValidateError(source_name, *line_nr, *col_nr,
            Exception::messages().format_message(
                MessageId::UNDEFINED_IDENTIFIER, *identifier_name));
    }
    catch(detail::UndefinedOperation const& exception) {
        String const& source_name = tree->source_name();

        String const* operation_name = boost::get_error_info<
            detail::ExceptionFunction>(exception);
        assert(operation_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        throw ValidateError(source_name, *line_nr, *col_nr,
            Exception::messages().format_message(
                MessageId::UNDEFINED_OPERATION, *operation_name));
    }
}


//! Execute the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to execute.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Validation (see validate(ScriptVertexPtr const&)).
  - Execution.
*/
void Interpreter::execute(
    ScriptVertexPtr const& tree)
{
    validate(tree);
    tree->Accept(_execute_visitor);
}


std::stack<std::shared_ptr<Argument>> Interpreter::stack()
{
    // std::stack<ResultType> result_types(_annotate_visitor.stack());
    // Stack values(_execute_visitor.stack());
    // assert(result_types.size() == values.size());
    // std::stack<std::tuple<ResultType, boost::any>> result;

    // for(size_t i = 0; i < _annotate_visitor.stack().size(); ++i) {
    //     result.push(std::make_tuple(result_types.top(), values.top()));
    //     result_types.pop();
    //     values.pop();
    // }

    // return result;

    return _execute_visitor.stack();
}


void Interpreter::clear_stack()
{
    _annotate_visitor.clear_stack();
    _execute_visitor.clear_stack();
}

} // namespace ranally
