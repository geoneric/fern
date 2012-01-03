#include <Python.h> // This one first, to get rid of preprocessor warnings.
#include "Ranally/Language/AlgebraParser.h"
#include <cassert>
#include <iostream>
#include "Python-ast.h"
#include <boost/format.hpp>
#include "dev_UnicodeUtils.h"
#include "Ranally/Python/Exception.h"



//   PyObject* type = PyObject_Type(id);
//   PyObject* string = PyObject_Str(type);
//   std::cout << PyString_AsString(string) << std::endl;



namespace {

void               writeExpressionNode (expr_ty const& expression,
                                        UnicodeString& xml);
void               writeStatementNodes (asdl_seq const* statements,
                                        UnicodeString& xml);



void writeNameNode(
  identifier const id,
  expr_context_ty const& /* context */,
  UnicodeString& xml)
{
  assert(PyString_Check(id));

  // TODO Handle Unicode. In modern Python, identifiers are Unicode strings.
  xml += "<Name>";
  xml += PyString_AsString(id);
  xml += "</Name>";

  // PyObject* bytes = PyUnicode_AsUTF8String(id);
  // assert(bytes);
  // xml += dev::decodeFromUTF8(PyBytes_AsString(bytes));
}



void writeNumberNode(
  object const& number,
  UnicodeString& xml)
{
  // TODO Handle all numeric types.
  // TODO Use types with a known size: int32, int64, float32, float64, etc.
  xml += "<Number>";

  // From the Python docs:
  //   Plain integers (also just called integers) are implemented using
  //   long in C, which gives them at least 32 bits of precision (sys.maxint is
  //   always set to the maximum plain integer value for the current platform,
  //   the minimum value is -sys.maxint - 1).
  if(PyInt_CheckExact(number)) {
    // The number object contains a C long value. The size of long is platform
    // dependent.
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
    // TODO Can we assume that the value fits in a long long? Otherwise, handle
    //      overflow.
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



void writeStringNode(
  string const string,
  UnicodeString& xml)
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



void writeExpressionsNode(
  asdl_seq const* expressions,
  UnicodeString& xml)
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
      writeExpressionNode(expression, xml);
    }

    xml += "</Expressions>";
  }
}



void writeCallNode(
  expr_ty const function,
  asdl_seq const* arguments,
  asdl_seq const* keywords,
  expr_ty const starargs,
  expr_ty const kwargs,
  UnicodeString& xml)
{
  assert(keywords == 0 || keywords->size == 0); // TODO Support keywords.
  assert(starargs == 0); // TODO
  assert(kwargs == 0); // TODO

  xml += "<Function>";
  assert(function->kind == Name_kind);
  writeNameNode(function->v.Name.id, function->v.Name.ctx, xml);
  writeExpressionsNode(arguments, xml);
  xml += "</Function>";
}



void writeUnaryOperatorNode(
  unaryop_ty const unaryOperator,
  expr_ty const operand,
  UnicodeString& xml)
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
      xml += "Add";
      break;
    }
    case USub: {
      xml += "Sub";
      break;
    }
    // Don't add a default clause! We want to hear from the compiler if we're
    // missing a case.
  }

  xml += "</Name>";
  xml += "<Expressions>";
  writeExpressionNode(operand, xml);
  xml += "</Expressions>";
  xml += "</Operator>";
}



void writeBinaryOperatorNode(
  expr_ty const leftOperand,
  operator_ty const binaryOperator,
  expr_ty const rightOperand,
  UnicodeString& xml)
{
  xml += "<Operator><Name>";

  switch(binaryOperator) {
    case Add: {
      xml += "Add";
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
    // Don't add a default clause! We want to hear from the compiler if we're
    // missing a case.
  }

  xml += "</Name>";
  xml += "<Expressions>";
  writeExpressionNode(leftOperand, xml);
  writeExpressionNode(rightOperand, xml);
  xml += "</Expressions>";
  xml += "</Operator>";
}



void writeBooleanOperatorNode(
  boolop_ty const booleanOperator,
  asdl_seq const* operands,
  UnicodeString& xml)
{
  xml += "<Operator><Name>";

  switch(booleanOperator) {
    case And: {
      xml += "And";
      break;
    }
    case Or: {
      xml += "Or";
      break;
    }
    // Don't add a default clause! We want to hear from the compiler if we're
    // missing a case.
  }

  xml += "</Name>";
  writeExpressionsNode(operands, xml);
  xml += "</Operator>";
}



void writeComparisonOperatorNode(
  expr_ty const leftOperand,
  asdl_int_seq const* operators,
  asdl_seq const* comparators,
  UnicodeString& xml)
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
      // TODO Exception. Unsupported operator. Is=7, IsNot=8, In=9, NotIn=10
      assert(false);
      break;
    }
  }

  xml += "</Name><Expressions>";
  writeExpressionNode(leftOperand, xml);
  writeExpressionNode(static_cast<expr_ty const>(asdl_seq_GET(comparators, 0)),
    xml);
  xml += "</Expressions></Operator>";
}



void writeExpressionNode(
  expr_ty const& expression,
  UnicodeString& xml)
{
  assert(expression);

  // 1-based linenumber.
  // 0-based column id.
  xml += (boost::format("<Expression line=\"%1%\" col=\"%2%\">")
    % expression->lineno
    % expression->col_offset).str().c_str();

  switch(expression->kind) {
    case Name_kind: {
      writeNameNode(expression->v.Name.id, expression->v.Name.ctx, xml);
      break;
    }
    case Num_kind: {
      writeNumberNode(expression->v.Num.n, xml);
      break;
    }
    case Str_kind: {
      writeStringNode(expression->v.Str.s, xml);
      break;
    }
    case Call_kind: {
      writeCallNode(expression->v.Call.func, expression->v.Call.args,
        expression->v.Call.keywords, expression->v.Call.starargs,
        expression->v.Call.kwargs, xml);
      break;
    }
    case UnaryOp_kind: {
      writeUnaryOperatorNode(expression->v.UnaryOp.op,
        expression->v.UnaryOp.operand, xml);
      break;
    }
    case BinOp_kind: {
      writeBinaryOperatorNode(expression->v.BinOp.left,
        expression->v.BinOp.op, expression->v.BinOp.right, xml);
      break;
    }
    case BoolOp_kind: {
      writeBooleanOperatorNode(expression->v.BoolOp.op,
        expression->v.BoolOp.values, xml);
      break;
    }
    case Compare_kind: {
      writeComparisonOperatorNode(expression->v.Compare.left,
        expression->v.Compare.ops, expression->v.Compare.comparators, xml);
      break;
    }
    case IfExp_kind:
    case Lambda_kind:
    case Dict_kind:
    case DictComp_kind:
    case GeneratorExp_kind:
    case Yield_kind:
    case Repr_kind:
    case Attribute_kind:
    case Subscript_kind:
    case List_kind:
    case ListComp_kind:
    case Set_kind:
    case SetComp_kind:
    case Tuple_kind: {
      // TODO exception
      bool implemented = false;
      assert(implemented);
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
void writeAssignmentNode(
  asdl_seq const* targets,
  expr_ty const& value,
  UnicodeString& xml)
{
  // Copy the value expression to one or more targets.
  // When there is more than one target, value should be an iterable.
  // For now we can limit the number of targets to one.

  assert(targets->size == 1); // TODO Error handling.
  expr_ty const target = static_cast<expr_ty const>(asdl_seq_GET(targets, 0));
  // We don't support attributeref, subscription and slicing.
  assert(target->kind == Name_kind); // TODO Error handling.

  xml += "<Assignment><Targets>";
  writeExpressionNode(target, xml);
  xml += "</Targets><Expressions>";
  writeExpressionNode(value, xml);
  xml += "</Expressions></Assignment>";
}



void writeIfNode(
  expr_ty const test,
  asdl_seq const* body,
  asdl_seq const* orelse,
  UnicodeString& xml)
{
  xml += "<If>";
  writeExpressionNode(test, xml);
  writeStatementNodes(body, xml);
  writeStatementNodes(orelse, xml);
  xml += "</If>";
}



void writeWhileNode(
  expr_ty const test,
  asdl_seq const* body,
  asdl_seq const* orelse,
  UnicodeString& xml)
{
  xml += "<While>";
  writeExpressionNode(test, xml);
  writeStatementNodes(body, xml);
  writeStatementNodes(orelse, xml);
  xml += "</While>";
}



void writeStatementNodes(
  asdl_seq const* statements,
  UnicodeString& xml)
{
  if(!statements || statements->size == 0) {
    xml += "<Statements/>";
  }
  else {
    xml += "<Statements>";

    // See Python-ast.h.
    for(int i = 0; i < statements->size; ++i) {
      stmt_ty const statement = static_cast<stmt_ty const>(
        asdl_seq_GET(statements, i));
      assert(statement);

      xml += "<Statement>";

      switch(statement->kind) {
        case Expr_kind: {
          writeExpressionNode(statement->v.Expr.value, xml);
          break;
        }
        case Assign_kind: {
          writeAssignmentNode(statement->v.Assign.targets,
            statement->v.Assign.value, xml);
          break;
        }
        case If_kind: {
          writeIfNode(statement->v.If.test, statement->v.If.body,
            statement->v.If.orelse, xml);
          break;
        }
        case While_kind: {
          writeWhileNode(statement->v.While.test, statement->v.While.body,
            statement->v.While.orelse, xml);
          break;
        }

        // TODO
        case Break_kind:
        case Continue_kind:
        case Assert_kind:
        case AugAssign_kind:

        case Print_kind:
        case Global_kind:
        case Pass_kind:

        case FunctionDef_kind:
        case ClassDef_kind:
        case Return_kind:
        case Delete_kind:
        case For_kind:
        case With_kind:
        case Raise_kind:
        case TryExcept_kind:
        case TryFinally_kind:
        case Import_kind:
        case ImportFrom_kind:
        case Exec_kind:
        {
          bool implemented = false;
          assert(implemented);
          break;
        }
      }

      xml += "</Statement>";
    }

    xml += "</Statements>";
  }
}



UnicodeString pythonAstToXml(
  mod_ty const ast,
  UnicodeString const& sourceName)
{
  assert(ast);

  UnicodeString xml;

  switch(ast->kind) {
    case Module_kind: {
      xml += (boost::format("<Ranally source=\"%1%\">")
        % dev::encodeInUTF8(sourceName)).str().c_str();
      writeStatementNodes(ast->v.Module.body, xml);
      xml += "</Ranally>";
      break;
    }
    case Expression_kind: // {
    //   writeExpressionNode(ast->v.Expression.body, xml);
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

  return xml;
}

} // Anonymous namespace



namespace ranally {
namespace language {

AlgebraParser::AlgebraParser()

  : dev::PythonClient()

{
  assert(dev::PythonClient::isInitialized());
}



AlgebraParser::~AlgebraParser()
{
}



// UnicodeString AlgebraParser::parse(
//   UnicodeString const& string,
//   UnicodeString const& fileName)
// {
//   // struct _node* PyParser_SimpleParseStringFlagsFilename(const char *str, const char *filename, int start, int flags)
// 
//   // struct _node* node = PyParser_SimpleParseStringFlagsFilename("string", "filename", 0, 0);
// 
//   return UnicodeString(string + fileName);
// }



//! Parse the script in \a string and return an XML document.
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .
  \todo      The Python memory arena is not freed.
*/
UnicodeString AlgebraParser::parseString(
  UnicodeString const& string)
{
  PyArena* arena = PyArena_New();
  assert(arena);

  UnicodeString result("<?xml version=\"1.0\"?>");

  mod_ty ast = PyParser_ASTFromString(
    dev::encodeInUTF8(string).c_str(), "", Py_file_input, 0, arena);

  if(!ast) {
    ranally::python::throwException();
  }

  result += pythonAstToXml(ast, "&lt;string&gt;");

  PyArena_Free(arena);
  arena = 0;

  return result;
}



UnicodeString AlgebraParser::parseFile(
  UnicodeString const& fileName)
{
  PyArena* arena = PyArena_New();
  assert(arena);

  std::string fileNameInUtf8(dev::encodeInUTF8(fileName));
  FILE* filePointer = fopen(fileNameInUtf8.c_str(), "r");

  if(filePointer == NULL) {
    throw std::runtime_error(("cannot open file " + fileNameInUtf8).c_str());
  }

  UnicodeString result("<?xml version=\"1.0\"?>");

  mod_ty ast = PyParser_ASTFromFile(filePointer, fileNameInUtf8.c_str(),
    Py_file_input, 0, 0, 0, 0, arena);

  if(!ast) {
    ranally::python::throwException();
  }

  result += pythonAstToXml(ast, fileName);

  PyArena_Free(arena);
  arena = 0;

  return result;
}

} // namespace language
} // namespace ranally

