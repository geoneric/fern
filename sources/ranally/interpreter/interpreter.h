#pragma once
#include "ranally/language/algebra_parser.h"
#include "ranally/language/annotate_visitor.h"
#include "ranally/language/script_vertex.h"
#include "ranally/language/validate_visitor.h"
#include "ranally/language/xml_parser.h"
#include "ranally/interpreter/execute_visitor.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Interpreter
{

    friend class InterpreterTest;

public:

                   Interpreter         ();

                   ~Interpreter        ()=default;

                   Interpreter         (Interpreter&&)=delete;

    Interpreter&   operator=           (Interpreter&&)=delete;

                   Interpreter         (Interpreter const&)=delete;

    Interpreter&   operator=           (Interpreter const&)=delete;

    ScriptVertexPtr parse_string       (String const& string) const;

    ScriptVertexPtr parse_file         (String const& filename) const;

    void           annotate            (ScriptVertexPtr const& tree);

    void           validate            (ScriptVertexPtr const& tree);

    void           execute             (ScriptVertexPtr const& tree);

    std::stack<std::tuple<ResultType, boost::any>> stack();

    void           clear_stack         ();

private:

    AlgebraParser  _algebra_parser;

    XmlParser      _xml_parser;

    AnnotateVisitor _annotate_visitor;

    ValidateVisitor _validate_visitor;

    ExecuteVisitor _execute_visitor;

};

} // namespace ranally
