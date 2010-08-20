#include <Python.h> // This one first, to get rid of preprocessor warnings.
#include <cassert>
#include <iostream>

#include "Python-ast.h"

#include "dev_UnicodeUtils.h"

#include "AlgebraParser.h"



namespace {

//   PyObject* type = PyObject_Type(id);
//   PyObject* string = PyObject_Str(type);
//   std::cout << PyString_AsString(string) << std::endl;

void writeNameNode(
         identifier const id,
         expr_context_ty const ctx,
         UnicodeString& xml)
{
  assert(PyString_Check(id));

  xml += PyString_AsString(id);


  // PyObject* bytes = PyUnicode_AsUTF8String(id);
  // assert(bytes);
  // xml += dev::decodeFromUTF8(PyBytes_AsString(bytes));
}



void writeExpressionNode(
         expr_ty const expression,
         UnicodeString& xml)
{
  assert(expression);

  xml += "<expression>";

  switch(expression->kind) {
    case Name_kind: {
      writeNameNode(expression->v.Name.id, expression->v.Name.ctx, xml);
      break;
    }
    default: {
      assert(false);
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
      default: {
        assert(false);
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

  // The result is a Python AST that needs to be converted to an XML string.
  // - How do we walk the tree? We cannot modify the node types. Is loki
  //   still useful? We can create a parallel visitor tree.

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
    default: {
      assert(false);
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



UnicodeString AlgebraParser::parseFile(
         UnicodeString const& fileName)
{
  PyArena* arena = PyArena_New();
  assert(arena);

  UnicodeString result(pythonAstToXml(PyParser_ASTFromString(
         "string", "file name", Py_file_input, 0, arena)));

  PyArena_Free(arena);
  arena = 0;

  return result;
}

} // namespace ranally

