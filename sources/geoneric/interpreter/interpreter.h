#pragma once
#include "geoneric/script/algebra_parser.h"
#include "geoneric/ast/core/module_vertex.h"
#include "geoneric/ast/visitor/annotate_visitor.h"
#include "geoneric/ast/visitor/validate_visitor.h"
#include "geoneric/ast/xml/xml_parser.h"
#include "geoneric/interpreter/execute_visitor.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Interpreter
{

public:

                   Interpreter         ();

                   ~Interpreter        ()=default;

                   Interpreter         (Interpreter&&)=delete;

    Interpreter&   operator=           (Interpreter&&)=delete;

                   Interpreter         (Interpreter const&)=delete;

    Interpreter&   operator=           (Interpreter const&)=delete;

    ModuleVertexPtr parse_string       (String const& string) const;

    ModuleVertexPtr parse_file         (String const& filename) const;

    void           annotate            (ModuleVertexPtr const& tree);

    void           validate            (ModuleVertexPtr const& tree);

    void           execute             (ModuleVertexPtr const& tree);

    std::stack<std::shared_ptr<Argument>>
                   stack               ();

    void           clear_stack         ();

private:

    OperationsPtr  _operations;

    AlgebraParser  _algebra_parser;

    XmlParser      _xml_parser;

    AnnotateVisitor _annotate_visitor;

    ValidateVisitor _validate_visitor;

    ExecuteVisitor _back_end;

};

} // namespace geoneric
