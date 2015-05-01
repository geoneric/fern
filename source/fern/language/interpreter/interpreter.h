// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/script/algebra_parser.h"
#include "fern/language/ast/core/module_vertex.h"
#include "fern/language/ast/visitor/annotate_visitor.h"
#include "fern/language/ast/visitor/validate_visitor.h"
#include "fern/language/ast/xml/xml_parser.h"
#include "fern/language/interpreter/execute_visitor.h"


namespace fern {
namespace language {

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

    ModuleVertexPtr parse_string       (std::string const& string) const;

    ModuleVertexPtr parse_file         (std::string const& filename) const;

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

} // namespace language
} // namespace fern
