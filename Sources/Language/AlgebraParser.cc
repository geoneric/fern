#include <Python.h> // This one first to get rid of preprocessor warnings.
#include <cassert>

#include "Python-ast.h"

#include "AlgebraParser.h"



namespace ranally {

AlgebraParser::AlgebraParser()

  : dev::PythonClient()

{
  assert(dev::PythonClient::isInitialized());
}



AlgebraParser::~AlgebraParser()
{
}



UnicodeString AlgebraParser::parse(
         UnicodeString const& string,
         UnicodeString const& fileName)
{
  // struct _node* PyParser_SimpleParseStringFlagsFilename(const char *str, const char *filename, int start, int flags)

  // struct _node* node = PyParser_SimpleParseStringFlagsFilename("string", "filename", 0, 0);

  return UnicodeString(string + fileName);
}



UnicodeString AlgebraParser::parseString(
         UnicodeString const& string)
{
  PyArena* arena = PyArena_New();
  assert(arena);

  {
    struct _mod * bla = PyParser_ASTFromString("string", "file name",
        Py_eval_input, 0, arena);
    assert(bla);
    assert(bla->kind == Expression_kind);
    _expr* body = bla->v.Expression.body;
    assert(body);
  }

  {
    struct _mod * bla = PyParser_ASTFromString("string", "file name",
           Py_file_input, 0, arena);
    assert(bla);
    assert(bla->kind == Module_kind);
    asdl_seq* body = bla->v.Module.body;
    assert(body);
  }

  // The result is a Python AST that needs to be converted to an XML string.
  // - How do we walk the tree? We cannot modify the node types. Is loki
  //   still useful? We can create a parallel visitor tree.

  PyArena_Free(arena);

  // Convert algebra to utf8.
  return UnicodeString(string);
}



UnicodeString AlgebraParser::parseFile(
         UnicodeString const& fileName)
{
  // Convert algebra to utf8.
  return UnicodeString(fileName);
}

} // namespace ranally

