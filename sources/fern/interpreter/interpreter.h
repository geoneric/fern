#pragma once
#include "fern/script/algebra_parser.h"
#include "fern/ast/core/module_vertex.h"
#include "fern/ast/visitor/annotate_visitor.h"
#include "fern/ast/visitor/validate_visitor.h"
#include "fern/ast/xml/xml_parser.h"
#include "fern/interpreter/execute_visitor.h"


namespace fern {

class DataSource;
class DataSync;


//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Interpreter
{

public:

    using DataSourceSymbolTable = SymbolTable<std::shared_ptr<DataSource>>;

    using DataSyncSymbolTable =  SymbolTable<std::shared_ptr<DataSync>>;

                   Interpreter         ();

                   ~Interpreter        ()=default;

                   Interpreter         (Interpreter&&)=delete;

    Interpreter&   operator=           (Interpreter&&)=delete;

                   Interpreter         (Interpreter const&)=delete;

    Interpreter&   operator=           (Interpreter const&)=delete;

    ModuleVertexPtr parse_string       (String const& string) const;

    ModuleVertexPtr parse_file         (String const& filename) const;

    void           annotate            (ModuleVertexPtr const& tree);

    void           annotate            (ModuleVertexPtr const& tree,
                                        DataSourceSymbolTable const&
                                            symbol_table);

    void           validate            (ModuleVertexPtr const& tree);

    void           validate            (ModuleVertexPtr const& tree,
                                        DataSourceSymbolTable const&
                                            symbol_table);

    void           execute             (ModuleVertexPtr const& tree);

    void           execute             (ModuleVertexPtr const& tree,
                                        DataSourceSymbolTable const&
                                            data_source_symbol_table,
                                        DataSyncSymbolTable const&
                                            data_sync_symbol_table);

    std::stack<std::shared_ptr<Argument>>
                   stack               ();

    void           clear_stack         ();

private:

    OperationsPtr  _operations;

    AlgebraParser  _algebra_parser;

    XmlParser      _xml_parser;

    AnnotateVisitor _annotate_visitor;

    ValidateVisitor _validate_visitor;

    std::shared_ptr<ExecuteVisitor> _back_end;

};

} // namespace fern
