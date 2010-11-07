#include <Python.h> // This one first, to get rid of preprocessor warnings.
#include <cassert>
#include <iostream>
#include "Python-ast.h"
#include <boost/format.hpp>

#include "dev_UnicodeUtils.h"

#include "AlgebraParser.h"



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

  if(PyInt_CheckExact(number)) {
    xml += "<Integer>";
    xml += (boost::format("%1%") % PyInt_AsLong(number)).str().c_str();
    xml += "</Integer>";
  }
  else if(PyLong_CheckExact(number)) {
    xml += "<Long>";
    xml += (boost::format("%1%") % PyLong_AsLong(number)).str().c_str();
    xml += "</Long>";
  }
  else if(PyFloat_CheckExact(number)) {
    xml += "<Double>";
    xml += (boost::format("%1%") % PyFloat_AsDouble(number)).str().c_str();
    xml += "</Double>";
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
    // Don't add a default clause! We want to hear from the compiler if we're missing
    // a case.
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
    // Don't add a default clause! We want to hear from the compiler if we're missing
    // a case.
  }

  xml += "</Name>";
  xml += "<Expressions>";
  writeExpressionNode(leftOperand, xml);
  writeExpressionNode(rightOperand, xml);
  xml += "</Expressions>";

  xml += "</Operator>";
}



void writeBooleanOperatorNode(
  boolop_ty booleanOperator,
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
    // Don't add a default clause! We want to hear from the compiler if we're missing
    // a case.
  }

  xml += "</Name>";
  writeExpressionsNode(operands, xml);
  xml += "</Operator>";
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
    case Lambda_kind:
    case IfExp_kind:
    case Dict_kind:
    case ListComp_kind:
    case GeneratorExp_kind:
    case Yield_kind:
    case Compare_kind:
    case Repr_kind:
    case Attribute_kind:
    case Subscript_kind:
    case List_kind:
    case Tuple_kind: {
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

  // expr_ty:
  //   BoolOp_kind=1, BinOp_kind=2, UnaryOp_kind=3, Lambda_kind=4,
  //   IfExp_kind=5, Dict_kind=6, ListComp_kind=7, GeneratorExp_kind=8,
  //   Yield_kind=9, Compare_kind=10, Call_kind=11, Repr_kind=12, Num_kind=13,
  //   Str_kind=14, Attribute_kind=15, Subscript_kind=16, Name_kind=17,
  //   List_kind=18, Tuple_kind=19

  assert(targets->size == 1); // TODO Error handling.
  expr_ty const target = static_cast<expr_ty const>(asdl_seq_GET(targets, 0));
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
        case Break_kind:
        case Continue_kind:
        case Assert_kind:
        case FunctionDef_kind:
        case ClassDef_kind:
        case Return_kind:
        case Delete_kind:
        case AugAssign_kind:
        case Print_kind:
        case For_kind:
        case While_kind:
        case With_kind:
        case Raise_kind:
        case TryExcept_kind:
        case TryFinally_kind:
        case Import_kind:
        case ImportFrom_kind:
        case Exec_kind:
        case Global_kind:
        case Pass_kind:
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
  mod_ty const ast)
{
  assert(ast);

  UnicodeString xml;

  switch(ast->kind) {
    case Module_kind: {
      xml += "<Ranally>";
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
      bool implemented = false;
      assert(implemented);
      break;
    }
  }

  return xml;
}

} // Anonymous namespace



namespace ranally {

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



UnicodeString AlgebraParser::parseString(
  UnicodeString const& string)
{
  PyArena* arena = PyArena_New();
  assert(arena);

  UnicodeString result("<?xml version=\"1.0\"?>");

  result += pythonAstToXml(PyParser_ASTFromString(
    dev::encodeInUTF8(string).c_str(), "", Py_file_input, 0, arena));

  // TODO Memory leak!
  // PyArena_Free(arena);
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

  // TODO Error handling. What if file does not exist.

  UnicodeString result("<?xml version=\"1.0\"?>");

  result += pythonAstToXml(PyParser_ASTFromFile(filePointer,
    fileNameInUtf8.c_str(), Py_file_input, 0, 0, 0, 0, arena));

  PyArena_Free(arena);
  arena = 0;

  return result;
}

} // namespace ranally

