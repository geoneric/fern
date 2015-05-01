// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/interpreter/interpreter.h"
#include <sstream>
#include "fern/core/io_error.h"
#include "fern/core/parse_error.h"
#include "fern/core/validate_error.h"
#include "fern/language/operation/std/operations.h"
#include "fern/language/ast/visitor/identify_visitor.h"
#ifndef NDEBUG
    #include "fern/language/ast/visitor/io_visitor.h"
#endif
#include "fern/language/ast/visitor/thread_visitor.h"
#include "fern/language/interpreter/data_sources.h"


namespace fern {
namespace language {

namespace {

SymbolTable<ExpressionType> expression_types(
    SymbolTable<std::shared_ptr<DataSource>> const& data_sources)
{
    SymbolTable<ExpressionType> result;

    if(!data_sources.empty()) {
        assert(data_sources.scope_level() == 1u);
        result.push_scope();

        for(auto data_source: data_sources.scope(1)) {
            result.add_value(data_source.first,
                data_source.second->expression_type());
        }
    }

    return result;
}


// DataSourceSymbolTable solve_read_calls(
//     ModuleVertexPtr const& tree)
// {
// }

} // Anonymous namespace


Interpreter::Interpreter()

    : _operations(operations()),
      _algebra_parser(),
      _xml_parser(),
      _annotate_visitor(_operations),
      _validate_visitor(),
      _back_end(std::make_shared<ExecuteVisitor>(_operations))

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
ModuleVertexPtr Interpreter::parse_string(
    std::string const& string) const
{
    ModuleVertexPtr script_vertex;

    try {
        script_vertex = _xml_parser.parse_string(_algebra_parser.parse_string(
            string));
    }
    catch(detail::ParseError const& exception) {
        std::string const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        std::string const* statement = boost::get_error_info<
            detail::ExceptionStatement>(exception);

        std::string const* message = boost::get_error_info<
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
        std::string const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        std::string const* construct = boost::get_error_info<
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
ModuleVertexPtr Interpreter::parse_file(
    std::string const& filename) const
{
    ModuleVertexPtr script_vertex;

    try {
        if(filename.empty()) {
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
        std::string const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        int const* errno_ = boost::get_error_info<boost::errinfo_errno>(
            exception);
        assert(errno_);

        throw IOError(*source_name, *errno_);
    }
    catch(detail::ParseError const& exception) {
        std::string const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        std::string const* statement = boost::get_error_info<
            detail::ExceptionStatement>(exception);

        std::string const* message = boost::get_error_info<
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
        std::string const* source_name = boost::get_error_info<
            detail::ExceptionSourceName>(exception);
        assert(source_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        std::string const* construct = boost::get_error_info<
            detail::ExceptionConstruct>(exception);
        assert(construct);

        throw ParseError(*source_name, *line_nr, *col_nr,
            Exception::messages().format_message(
                MessageId::UNSUPPORTED_LANGUAGE_CONSTRUCT, *construct));
    }

    return script_vertex;
}


void Interpreter::annotate(
    ModuleVertexPtr const& tree)
{
    annotate(tree, DataSourceSymbolTable());
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
    ModuleVertexPtr const& tree,
    DataSourceSymbolTable const& symbol_table)
{
    assert(symbol_table.empty() || symbol_table.scope_level() == 1u);

    // // Replace read calls by undefined identifiers and return a symbol
    // // table linking the identifiers to data sources. This way, we can
    // // treat undefined identifiers and read calls in a similar fashion.
    // DataSourceSymbolTable read_calls_symbol_table = solve_read_calls(tree);

    ThreadVisitor thread_visitor;
    tree->Accept(thread_visitor);

    IdentifyVisitor identify_visitor;
    tree->Accept(identify_visitor);

    _annotate_visitor.add_global_symbols(expression_types(symbol_table));
    tree->Accept(_annotate_visitor);
}


void Interpreter::validate(
    ModuleVertexPtr const& tree)
{
    validate(tree, DataSourceSymbolTable());
}


//! Validate the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to validate.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Annotation (see annotate(ModuleVertexPtr const&, DataSourceSymbolTable
    const&)).
  - Validation.
*/
void Interpreter::validate(
    ModuleVertexPtr const& tree,
    DataSourceSymbolTable const& data_source_symbol_table)
{
    try {
        annotate(tree, data_source_symbol_table);
        tree->Accept(_validate_visitor);
    }
    catch(detail::UndefinedIdentifier const& exception) {
        std::string const& source_name = tree->source_name();

        std::string const* identifier_name = boost::get_error_info<
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
        std::string const& source_name = tree->source_name();

        std::string const* operation_name = boost::get_error_info<
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
    catch(detail::WrongNumberOfArguments const& exception) {
        std::string const& source_name = tree->source_name();

        std::string const* operation_name = boost::get_error_info<
            detail::ExceptionFunction>(exception);
        assert(operation_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        size_t const* required_nr_arguments = boost::get_error_info<
            detail::ExceptionRequiredNrArguments>(exception);
        size_t const* provided_nr_arguments = boost::get_error_info<
            detail::ExceptionProvidedNrArguments>(exception);

        throw ValidateError(source_name, *line_nr, *col_nr,
            Exception::messages().format_message(
                MessageId::WRONG_NUMBER_OF_ARGUMENTS, *operation_name,
                *required_nr_arguments, *provided_nr_arguments));
    }
    catch(detail::WrongTypeOfArgument const& exception) {
        std::string const& source_name = tree->source_name();

        std::string const* operation_name = boost::get_error_info<
            detail::ExceptionFunction>(exception);
        assert(operation_name);

        long const* line_nr = boost::get_error_info<
            detail::ExceptionLineNr>(exception);
        assert(line_nr);

        long const* col_nr = boost::get_error_info<
            detail::ExceptionColNr>(exception);
        assert(col_nr);

        size_t const* argument_id = boost::get_error_info<
            detail::ExceptionArgumentId>(exception);
        std::string const* required_argument_types = boost::get_error_info<
            detail::ExceptionRequiredArgumentTypes>(exception);
        std::string const* provided_argument_types = boost::get_error_info<
            detail::ExceptionProvidedArgumentTypes>(exception);

        throw ValidateError(source_name, *line_nr, *col_nr,
            Exception::messages().format_message(
                MessageId::WRONG_TYPE_OF_ARGUMENT, *argument_id,
                *operation_name, *required_argument_types,
                *provided_argument_types));
    }
}


void Interpreter::execute(
    ModuleVertexPtr const& tree)
{
    execute(tree, DataSourceSymbolTable(), DataSyncSymbolTable());
}


//! Execute the model in \a tree.
/*!
  \param     tree Syntax tree containing the model to execute.
  \exception .

  This is a top level operation. It is assumed here that \a tree is the
  result of parsing a script, without further processing.

  The folowing steps are performed:
  - Validation (see validate(ModuleVertexPtr const&)).
  - Execution.
*/
void Interpreter::execute(
    ModuleVertexPtr const& tree,
    DataSourceSymbolTable const& data_source_symbol_table,
    DataSyncSymbolTable const& data_sync_symbol_table)
{
    validate(tree, data_source_symbol_table);

// TODO This doesn't work yet. We must handle read calls or forbid them.
// #ifndef NDEBUG
//     // By now, the expression types of all outputs should be known. Otherwise
//     // annotation and/or validation made a mistake.
//     IOVisitor visitor;
//     tree->Accept(visitor);
//     for(auto const& output: visitor.outputs()) {
//         for(auto const& expression_type: output->expression_types()) {
//             if(!expression_type.fixed()) {
//                 std::cout << output->name() << std::endl;
//             }
//             assert(expression_type.fixed());
//         }
//     }
// #endif

    // TODO Catch exceptions. Out of memory, ..., ...
    _back_end->set_data_source_symbols(data_source_symbol_table);
    _back_end->set_data_sync_symbols(data_sync_symbol_table);
    tree->Accept(*_back_end);

    // ExecutionManager manager(*_back_end, data_source_symbol_table);
    // manager.run();
}


std::stack<std::shared_ptr<Argument>> Interpreter::stack()
{
    // std::stack<ResultType> expression_types(_annotate_visitor.stack());
    // Stack values(_back_end->stack());
    // assert(expression_types.size() == values.size());
    // std::stack<std::tuple<ResultType, boost::any>> result;

    // for(size_t i = 0; i < _annotate_visitor.stack().size(); ++i) {
    //     result.push(std::make_tuple(expression_types.top(), values.top()));
    //     expression_types.pop();
    //     values.pop();
    // }

    // return result;

    return _back_end->stack();
}


void Interpreter::clear_stack()
{
    _annotate_visitor.clear_stack();
    _back_end->clear_stack();
}

} // namespace language
} // namespace fern
