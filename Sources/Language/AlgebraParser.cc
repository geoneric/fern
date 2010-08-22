#include <Python.h> // This one first, to get rid of preprocessor warnings.
#include <cassert>
#include <iostream>
#include "Python-ast.h"
#include <boost/format.hpp>

#include "dev_UnicodeUtils.h"

#include "AlgebraParser.h"



namespace {

//   PyObject* type = PyObject_Type(id);
//   PyObject* string = PyObject_Str(type);
//   std::cout << PyString_AsString(string) << std::endl;

void writeNameNode(
         identifier const id,
         expr_context_ty const /* context */,
         UnicodeString& xml)
{
  assert(PyString_Check(id));

  xml += "<name>";
  xml += PyString_AsString(id);
  xml += "</name>";

  // PyObject* bytes = PyUnicode_AsUTF8String(id);
  // assert(bytes);
  // xml += dev::decodeFromUTF8(PyBytes_AsString(bytes));
}



void writeExpressionNode(
         expr_ty const expression,
         UnicodeString& xml)
{
  assert(expression);

  // 1-based linenumber.
  // 0-based column id.
  xml += (boost::format("<expression line=\"%1%\" col=\"%2%\">")
         % expression->lineno
         % expression->col_offset).str().c_str();

  switch(expression->kind) {
    case Name_kind: {
      writeNameNode(expression->v.Name.id, expression->v.Name.ctx, xml);
      break;
    }
    case BoolOp_kind:
    case BinOp_kind:
    case UnaryOp_kind:
    case Lambda_kind:
    case IfExp_kind:
    case Dict_kind:
    case ListComp_kind:
    case GeneratorExp_kind:
    case Yield_kind:
    case Compare_kind:
    case Call_kind:
    case Repr_kind:
    case Num_kind:
    case Str_kind:
    case Attribute_kind:
    case Subscript_kind:
    case List_kind:
    case Tuple_kind: {
      bool implemented = false;
      assert(implemented);
      break;
    }
  }

  xml += "</expression>";
}



void writeModuleNode(
         asdl_seq const* statements,
         UnicodeString& xml)
{
  assert(statements);

  xml += "<module>";

  for(int i = 0; i < statements->size; ++i) {
    stmt_ty const statement = static_cast<stmt_ty const>(
         asdl_seq_GET(statements, i));
    assert(statement);

    switch(statement->kind) {
      case Expr_kind: {
        writeExpressionNode(statement->v.Expr.value, xml);
        break;
      }
      case FunctionDef_kind:
      case ClassDef_kind:
      case Return_kind:
      case Delete_kind:
      case Assign_kind:
      case AugAssign_kind:
      case Print_kind:
      case For_kind:
      case While_kind:
      case If_kind:
      case With_kind:
      case Raise_kind:
      case TryExcept_kind:
      case TryFinally_kind:
      case Assert_kind:
      case Import_kind:
      case ImportFrom_kind:
      case Exec_kind:
      case Global_kind:
      case Pass_kind:
      case Break_kind:
      case Continue_kind: {
        bool implemented = false;
        assert(implemented);
        break;
      }
    }
  }

  xml += "</module>";
}



UnicodeString pythonAstToXml(
         mod_ty const ast)
{
  assert(ast);

  UnicodeString xml;

  switch(ast->kind) {
    case Module_kind: {
      writeModuleNode(ast->v.Module.body, xml);
      break;
    }
    case Expression_kind: {
      writeExpressionNode(ast->v.Expression.body, xml);
      break;
    }
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
//          UnicodeString const& string,
//          UnicodeString const& fileName)
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

  UnicodeString result(pythonAstToXml(PyParser_ASTFromString(
         dev::encodeInUTF8(string).c_str(), "", Py_file_input, 0, arena)));

  PyArena_Free(arena);
  arena = 0;

  return result;
}



/// UnicodeString AlgebraParser::parseFile(
///          UnicodeString const& fileName)
/// {
///   PyArena* arena = PyArena_New();
///   assert(arena);
/// 
///   UnicodeString result(pythonAstToXml(PyParser_ASTFromString(
///          "string", "file name", Py_file_input, 0, arena)));
/// 
///   PyArena_Free(arena);
///   arena = 0;
/// 
///   return result;
/// }

} // namespace ranally

