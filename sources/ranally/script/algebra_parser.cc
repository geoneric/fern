#include <Python.h> // This one first, to get rid of preprocessor warnings.
#include "ranally/script/algebra_parser.h"
#include <cassert>
#include <iostream>
#include "Python-ast.h"
#include "ranally/core/exception.h"
#include "ranally/python/exception.h"


//  PyObject* type = PyObject_Type(id);
//  PyObject* string = PyObject_Str(type);
//  std::cout << PyString_AsString(string) << std::endl;


namespace {

void               write_expression_node(
                                        expr_ty const& expression,
                                        ranally::String& xml);
void               write_expression_nodes(
                                        asdl_seq const* expressions,
                                        ranally::String& xml);
void               write_statement_nodes(
                                        asdl_seq const* statements,
                                        ranally::String& xml);


void throw_unsupported_language_construct(
    long line_nr,
    long col_nr,
    ranally::String const& kind)
{
    BOOST_THROW_EXCEPTION(ranally::detail::UnsupportedLanguageConstruct()
        << ranally::detail::ExceptionConstruct(kind)
        << ranally::detail::ExceptionLineNr(line_nr)
        << ranally::detail::ExceptionColNr(col_nr)
    );
}


void write_identifier_node(
    identifier const identifier,
    ranally::String& xml)
{
    assert(PyString_Check(identifier));

    // TODO Handle Unicode. In modern Python, identifiers are Unicode strings.
    xml += "<Name>";
    xml += PyString_AsString(identifier);
    xml += "</Name>";

    // PyObject* bytes = PyUnicode_AsUTF8String(identifier);
    // assert(bytes);
    // xml += dev::decodeFromUTF8(PyBytes_AsString(bytes));
}


void write_name_node(
    identifier const identifier,
    expr_context_ty const& /* context */,
    ranally::String& xml)
{
    write_identifier_node(identifier, xml);
}


void write_number_node(
    object const& number,
    ranally::String& xml)
{
    // TODO Handle all numeric types.
    // TODO Use types with a known size: int32, int64, float32, float64, etc.
    xml += "<Number>";

    // From the Python docs:
    //   Plain integers (also just called integers) are implemented using
    //   long in C, which gives them at least 32 bits of precision
    //   (sys.maxint is always set to the maximum plain integer value for the
    //   current platform, the minimum value is -sys.maxint - 1).
    if(PyInt_CheckExact(number)) {
        // The number object contains a C long value. The size of long is
        // platform dependent.
        xml += "<Integer>";
        xml += "<Size>";
        xml += (boost::format("%1%") % (sizeof(long) * 8)).str().c_str();
        xml += "</Size>";
        xml += "<Value>";
        xml += (boost::format("%1%") % PyInt_AsLong(number)).str().c_str();
        xml += "</Value>";
        xml += "</Integer>";
    }
    else if(PyLong_CheckExact(number)) {
        // TODO Can we assume that the value fits in a long long? Otherwise,
        // handle overflow.
        xml += "<Integer>";
        xml += "<Size>";
        xml += (boost::format("%1%") % (sizeof(long long) * 8)).str().c_str();
        xml += "</Size>";
        xml += "<Value>";
        xml += (boost::format("%1%") % PyLong_AsLongLong(number)).str().c_str();
        xml += "</Value>";
        xml += "</Integer>";
    }
    else if(PyFloat_CheckExact(number)) {
        // TODO What is the size of a Python float?
        xml += "<Float>";
        xml += "<Size>";
        xml += (boost::format("%1%") % (sizeof(double) * 8)).str().c_str();
        xml += "</Size>";
        xml += "<Value>";
        xml += (boost::format("%1%") % PyFloat_AsDouble(number)).str().c_str();
        xml += "</Value>";
        xml += "</Float>";
    }
    else {
        // TODO Error handling.
        assert(false);
    }

    xml += "</Number>";
}


void write_string_node(
    string const string,
    ranally::String& xml)
{
    // TODO Verify the string is encoded as UTF8.
    // TODO Only support Unicode strings? Convert on the fly when it's not?
    assert(PyString_Check(string));

    if(PyString_Size(string) == 0) {
        xml += "<String/>";
    }
    else {
        xml += "<String>";
        xml += PyString_AsString(string);
        xml += "</String>";
    }
}


void write_expressions_node(
    asdl_seq const* expressions,
    ranally::String& xml)
{
    if(expressions == 0) {
        xml += "<Expressions/>";
    }
    else {
        assert(expressions->size > 0);
        xml += "<Expressions>";

        for(int i = 0; i < expressions->size; ++i) {
            expr_ty const expression = static_cast<expr_ty const>(
                asdl_seq_GET(expressions, i));
            write_expression_node(expression, xml);
        }

        xml += "</Expressions>";
    }
}


void write_call_node(
    expr_ty const function,
    asdl_seq const* arguments,
    asdl_seq const* keywords,
    expr_ty const starargs,
    expr_ty const kwargs,
    ranally::String& xml)
{
    assert(keywords == 0 || keywords->size == 0); // TODO Support keywords.
    assert(starargs == 0); // TODO
    assert(kwargs == 0); // TODO

    xml += "<Function>";
    assert(function->kind == Name_kind);
    write_name_node(function->v.Name.id, function->v.Name.ctx, xml);
    write_expressions_node(arguments, xml);
    xml += "</Function>";
}


void write_unary_operator_node(
    unaryop_ty const unaryOperator,
    expr_ty const operand,
    ranally::String& xml)
{
    xml += "<Operator><Name>";

    switch(unaryOperator) {
        case Invert: {
            xml += "Invert";
            break;
        }
        case Not: {
            xml += "Not";
            break;
        }
        case UAdd: {
            xml += "add";
            break;
        }
        case USub: {
            xml += "Sub";
            break;
        }
        // Don't add a default clause! We want to hear from the compiler if
        // we're missing a case.
    }

    xml += "</Name>";
    xml += "<Expressions>";
    write_expression_node(operand, xml);
    xml += "</Expressions>";
    xml += "</Operator>";
}


void write_binary_operator_node(
    expr_ty const left_operand,
    operator_ty const binary_operator,
    expr_ty const right_operand,
    ranally::String& xml)
{
    xml += "<Operator><Name>";

    switch(binary_operator) {
        case Add: {
            xml += "add";
            break;
        }
        case Sub: {
            xml += "Sub";
            break;
        }
        case Mult: {
            xml += "Mult";
            break;
        }
        case Div: {
            xml += "Div";
            break;
        }
        case Mod: {
            xml += "Mod";
            break;
        }
        case Pow: {
            xml += "Pow";
            break;
        }
        case LShift: {
            xml += "LShift";
            break;
        }
        case RShift: {
            xml += "RShift";
            break;
        }
        case BitOr: {
            xml += "BitOr";
            break;
        }
        case BitXor: {
            xml += "BitXor";
            break;
        }
        case BitAnd: {
            xml += "BitAnd";
            break;
        }
        case FloorDiv: {
            xml += "FloorDiv";
            break;
        }
        // Don't add a default clause! We want to hear from the compiler if
        // we're missing a case.
    }

    xml += "</Name>";
    xml += "<Expressions>";
    write_expression_node(left_operand, xml);
    write_expression_node(right_operand, xml);
    xml += "</Expressions>";
    xml += "</Operator>";
}


void write_boolean_operator_node(
    boolop_ty const boolean_operator,
    asdl_seq const* operands,
    ranally::String& xml)
{
    xml += "<Operator><Name>";

    switch(boolean_operator) {
        case And: {
            xml += "And";
            break;
        }
        case Or: {
            xml += "Or";
            break;
        }
        // Don't add a default clause! We want to hear from the compiler if
        // we're missing a case.
    }

    xml += "</Name>";
    write_expressions_node(operands, xml);
    xml += "</Operator>";
}


void write_comparison_operator_node(
    expr_ty const left_operand,
    asdl_int_seq const* operators,
    asdl_seq const* comparators,
    ranally::String& xml)
{
    // http://docs.python.org/reference/expressions.html#notin
    // x < y <= z is equivalent to x < y and y <= z

    assert(operators->size == 1); // TODO
    assert(operators->size == comparators->size);

    xml += "<Operator><Name>";

    switch(operators->elements[0]) {
        case Eq: {
            xml += "Eq";
            break;
        }
        case NotEq: {
            xml += "NotEq";
            break;
        }
        case Lt: {
            xml += "Lt";
            break;
        }
        case LtE: {
            xml += "LtE";
            break;
        }
        case Gt: {
            xml += "Gt";
            break;
        }
        case GtE: {
            xml += "GtE";
            break;
        }
        default: {
            // TODO Exception. Unsupported operator. Is=7, IsNot=8, In=9,
            //      NotIn=10
            assert(false);
            break;
        }
    }

    xml += "</Name><Expressions>";
    write_expression_node(left_operand, xml);
    write_expression_node(static_cast<expr_ty const>(
        asdl_seq_GET(comparators, 0)), xml);
    xml += "</Expressions></Operator>";
}


void write_slice_node(
    slice_ty const slice,
    ranally::String& xml)
{
    // TODO Raise exception.
    assert(slice->kind == Index_kind);

    switch(slice->kind) {
        case Ellipsis_kind: {
            break;
        }
        case Slice_kind: {
            break;
        }
        case ExtSlice_kind: {
            break;
        }
        case Index_kind: {
            write_expression_node(slice->v.Index.value, xml);
            break;
        }
    }
}


void write_subscript_node(
    expr_ty const expression,
    slice_ty const slice,
    expr_context_ty const /* context */,
    ranally::String& xml)
{
    // expression is the expression being subscripted.
    xml += "<Subscript>";
    write_expression_node(expression, xml);
    write_slice_node(slice, xml);
    xml += "</Subscript>";
}


void write_expression_node(
    expr_ty const& expression,
    ranally::String& xml)
{
    assert(expression);

    // 1-based linenumber.
    // 0-based column id.
    long line_nr = expression->lineno;
    long col_nr = expression->col_offset;
    xml += (boost::format("<Expression line=\"%1%\" col=\"%2%\">")
        % line_nr
        % col_nr).str().c_str();

    switch(expression->kind) {
        case Name_kind: {
            write_name_node(expression->v.Name.id, expression->v.Name.ctx, xml);
            break;
        }
        case Num_kind: {
            write_number_node(expression->v.Num.n, xml);
            break;
        }
        case Str_kind: {
            write_string_node(expression->v.Str.s, xml);
            break;
        }
        case Call_kind: {
            write_call_node(expression->v.Call.func, expression->v.Call.args,
                expression->v.Call.keywords, expression->v.Call.starargs,
                expression->v.Call.kwargs, xml);
            break;
        }
        case UnaryOp_kind: {
            write_unary_operator_node(expression->v.UnaryOp.op,
                expression->v.UnaryOp.operand, xml);
            break;
        }
        case BinOp_kind: {
            write_binary_operator_node(expression->v.BinOp.left,
                expression->v.BinOp.op, expression->v.BinOp.right, xml);
            break;
        }
        case BoolOp_kind: {
            write_boolean_operator_node(expression->v.BoolOp.op,
                expression->v.BoolOp.values, xml);
            break;
        }
        case Compare_kind: {
            write_comparison_operator_node(expression->v.Compare.left,
                expression->v.Compare.ops, expression->v.Compare.comparators,
                xml);
            break;
        }
        case Subscript_kind: {
            write_subscript_node(expression->v.Subscript.value,
                expression->v.Subscript.slice, expression->v.Subscript.ctx,
                xml);
            break;
        }
        case IfExp_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "if");
            break;
        }
        case Lambda_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "lambda");
            break;
        }
        case Dict_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "dictionary");
            break;
        }
        case DictComp_kind: {
            throw_unsupported_language_construct(line_nr, col_nr,
                "dictionary comprehension");
            break;
        }
        case GeneratorExp_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "generator");
            break;
        }
        case Yield_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "yield");
            break;
        }
        case Repr_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "repr");
            break;
        }
        case Attribute_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "attribute");
            break;
        }
        case List_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "list");
            break;
        }
        case ListComp_kind: {
            throw_unsupported_language_construct(line_nr, col_nr,
                "list comprehension");
            break;
        }
        case Set_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "set");
            break;
        }
        case SetComp_kind: {
            throw_unsupported_language_construct(line_nr, col_nr,
                "set comprehension");
            break;
        }
        case Tuple_kind: {
            throw_unsupported_language_construct(line_nr, col_nr, "tuple");
            break;
        }
    }

    xml += "</Expression>";
}


//!
/*!
  \param     targets One or more targets of the value or values passed in.
  \param     value One or more values to assign to the targets.
  \return    .
  \exception .
  \warning   .
  \sa        .
*/
void write_assignment_node(
    asdl_seq const* targets,
    expr_ty const& value,
    ranally::String& xml)
{
    // Copy the value expression to one or more targets.
    // When there is more than one target, value should be an iterable.
    // For now we can limit the number of targets to one.

    assert(targets->size == 1); // TODO Error handling.
    expr_ty const target = static_cast<expr_ty const>(asdl_seq_GET(targets, 0));
    // We don't support attributeref, subscription and slicing.
    assert(target->kind == Name_kind); // TODO Error handling.

    xml += "<Assignment>";
    write_expression_node(target, xml);
    write_expression_node(value, xml);
    xml += "</Assignment>";
}


void write_if_node(
    expr_ty const test,
    asdl_seq const* body,
    asdl_seq const* orelse,
    ranally::String& xml)
{
    xml += "<If>";
    write_expression_node(test, xml);
    write_statement_nodes(body, xml);
    write_statement_nodes(orelse, xml);
    xml += "</If>";
}


void write_while_node(
    expr_ty const test,
    asdl_seq const* body,
    asdl_seq const* orelse,
    ranally::String& xml)
{
    xml += "<While>";
    write_expression_node(test, xml);
    write_statement_nodes(body, xml);
    write_statement_nodes(orelse, xml);
    xml += "</While>";
}


void write_arguments_node(
    arguments_ty const arguments,
    ranally::String& xml)
{
    // TODO Throw exception in case vararg, kwarg, or defaults are set.
    write_expression_nodes(arguments->args, xml);
}


void write_function_definition_node(
    identifier const name,
    arguments_ty const arguments,
    asdl_seq const* body,
    asdl_seq const* /* decorator_list */,
    ranally::String& xml)
{
    // TODO Exception when decorator_list is not empty.
    xml += "<FunctionDefinition>";
    write_identifier_node(name, xml);
    write_arguments_node(arguments, xml);
    write_statement_nodes(body, xml);
    xml += "</FunctionDefinition>";
}


void write_return_node(
    expr_ty const value,
    ranally::String& xml)
{
    xml += "<Return>";
    write_expression_node(value, xml);
    xml += "</Return>";
}


// void writePrintFunctionNode(
//     expr_ty const dest,
//     asdl_seq const* values,
//     ranally::String& xml)
// {
//     // assert(!values || values->size == 0); // built-in print specific.
//     // assert(values->size == 1);
//     // 1-based linenumber.
//     // 0-based column id.
//     xml += (boost::format("<Expression line=\"%1%\" col=\"%2%\">")
//         % expression->lineno
//         % expression->col_offset);
//     xml += "<Function>";
//     xml += "<Name>";
//     xml += "print";
//     xml += "</Name>";
//     write_expressions_node(values, xml);
//     xml += "</Function>";
//     xml += "</Expression>";
// }


void write_expression_nodes(
    asdl_seq const* expressions,
    ranally::String& xml)
{
    if(!expressions || asdl_seq_LEN(expressions) == 0) {
        xml += "<Expressions/>";
    }
    else {
        xml += "<Expressions>";

        for(int i = 0; i < asdl_seq_LEN(expressions); ++i) {
            expr_ty const expression = static_cast<expr_ty const>(
                asdl_seq_GET(expressions, i));
            assert(expression);

            write_expression_node(expression, xml);

        }

        xml += "</Expressions>";
    }
}


void write_statement_nodes(
    asdl_seq const* statements,
    ranally::String& xml)
{
    if(!statements || asdl_seq_LEN(statements) == 0) {
        xml += "<Statements/>";
    }
    else {
        xml += "<Statements>";

        // See Python-ast.h.
        for(int i = 0; i < asdl_seq_LEN(statements); ++i) {
            stmt_ty const statement = static_cast<stmt_ty const>(
                asdl_seq_GET(statements, i));
            assert(statement);

            // 1-based linenumber.
            // 0-based column id.
            long line_nr = statement->lineno;
            long col_nr = statement->col_offset;
            // TODO Add line/col to XML.
            // xml += "<Statement>";
            xml += (boost::format("<Statement line=\"%1%\" col=\"%2%\">")
                % line_nr
                % col_nr).str().c_str();

            switch(statement->kind) {
                case Expr_kind: {
                    write_expression_node(statement->v.Expr.value, xml);
                    break;
                }
                case Assign_kind: {
                    write_assignment_node(statement->v.Assign.targets,
                        statement->v.Assign.value, xml);
                    break;
                }
                case If_kind: {
                    write_if_node(statement->v.If.test, statement->v.If.body,
                        statement->v.If.orelse, xml);
                    break;
                }
                case While_kind: {
                    write_while_node(statement->v.While.test,
                        statement->v.While.body, statement->v.While.orelse,
                        xml);
                    break;
                }
                case FunctionDef_kind: {
                    write_function_definition_node(
                        statement->v.FunctionDef.name,
                        statement->v.FunctionDef.args,
                        statement->v.FunctionDef.body,
                        statement->v.FunctionDef.decorator_list, xml);
                    break;
                }
                case Return_kind: {
                    write_return_node(statement->v.Return.value, xml);
                    break;
                }
                case Print_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "print");
                    break;
                }

                // {
                //   writePrintFunctionNode(statement->v.Print.dest,
                //     statement->v.Print.values, xml);
                //   break;
                // }

                case Break_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "break");
                    break;
                }
                case Continue_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "continue");
                    break;
                }
                case Assert_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "assert");
                    break;
                }
                case AugAssign_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "augmented assign");
                    break;
                }

                case Global_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "global");
                    break;
                }
                case Pass_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "pass");
                    break;
                }

                case ClassDef_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "define class");
                    break;
                }
                case Delete_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "delete");
                    break;
                }
                case For_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "for");
                    break;
                }
                case With_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "with");
                    break;
                }
                case Raise_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "raise");
                    break;
                }
                case TryExcept_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "try/except");
                    break;
                }
                case TryFinally_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "try/finally");
                    break;
                }
                case Import_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "import");
                    break;
                }
                case ImportFrom_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "import from");
                    break;
                }
                case Exec_kind: {
                    throw_unsupported_language_construct(line_nr, col_nr,
                        "exec");
                    break;
                }
            }

            xml += "</Statement>";
        }

        xml += "</Statements>";
    }
}


// <string> -> "&lt;string&gt;"
ranally::String escape(
    ranally::String const& string)
{
    return ranally::String(string)
        .replace("&" , "&amp;")  // This one first!
        .replace("<" , "&lt;")
        .replace(">" , "&gt;")
        .replace("\"", "&quot;")
        .replace("'" , "&apos;")
        ;
}


ranally::String python_ast_to_xml(
    mod_ty const ast,
    ranally::String const& source_name)
{
    assert(ast);

    ranally::String xml;

    try {
        switch(ast->kind) {
            case Module_kind: {
                xml += (boost::format("<Ranally source=\"%1%\">")
                    % escape(source_name).encode_in_utf8()).str().c_str();
                write_statement_nodes(ast->v.Module.body, xml);
                xml += "</Ranally>";
                break;
            }
            case Expression_kind: // {
            //   write_expression_node(ast->v.Expression.body, xml);
            //   break;
            // }
            case Interactive_kind:
            case Suite_kind: {
                // TODO Error message.
                bool implemented = false;
                assert(implemented);
                break;
            }
        }
    }
    catch(ranally::detail::Exception& exception) {
        exception
            << ranally::detail::ExceptionSourceName(source_name)
            ;
        throw;
    }

    return xml;
}


//! A small class to make creation and deletion of PyArena's easier.
/*!
    Whenever the SmartArena goes out of scope, the layered PyArena pointer
    is freed.
*/
class SmartArena
{

public:

    SmartArena()
        : _arena(PyArena_New())
    {
        assert(_arena);
    }

    ~SmartArena()
    {
        PyArena_Free(_arena);
    }

    SmartArena(SmartArena&&)=delete;

    SmartArena& operator=(SmartArena&&)=delete;

    SmartArena(SmartArena&)=delete;

    SmartArena& operator=(SmartArena&)=delete;

    PyArena* arena()
    {
        return _arena;
    }

private:

    PyArena* _arena;

};

} // Anonymous namespace


namespace ranally {

//! Construct an AlgebraParser instance.
/*!
*/
AlgebraParser::AlgebraParser()

    : python::Client()

{
    assert(python::Client::is_initialized());
}


//! Parse the script in \a string and return an XML document.
/*!
  \param     string String containing the script to parse.
  \return    String containing the script as an XML.
  \exception .
  \warning   .
  \sa        .
*/
String AlgebraParser::parse_string(
    String const& string) const
{
    SmartArena smart_arena;
    mod_ty ast = PyParser_ASTFromString(
        string.encode_in_utf8().c_str(), "", Py_file_input, 0,
        smart_arena.arena());

    if(!ast) {
        python::throw_exception("<string>");
    }

    return String("<?xml version=\"1.0\"?>") + python_ast_to_xml(ast,
        "<string>");
}


//! Parse the script in file \a filename and return an XML document.
/*!
  \param     filename Name of file containing the script to parse.
  \return    String containing the script as an XML.
  \exception .
  \warning   .
  \sa        .
*/
String AlgebraParser::parse_file(
    String const& filename) const
{
    SmartArena smart_arena;
    std::string filename_in_utf8(filename.encode_in_utf8());
    FILE* file_pointer = fopen(filename_in_utf8.c_str(), "r");

    if(file_pointer == NULL) {
        BOOST_THROW_EXCEPTION(detail::FileOpenError()
            << boost::errinfo_errno(errno)
            << detail::ExceptionSourceName(filename));
    }

    mod_ty ast = PyParser_ASTFromFile(file_pointer, filename_in_utf8.c_str(),
        Py_file_input, 0, 0, 0, 0, smart_arena.arena());

    if(!ast) {
        python::throw_exception(filename);
    }

    return String("<?xml version=\"1.0\"?>") + python_ast_to_xml(ast, filename);
}

} // namespace ranally
