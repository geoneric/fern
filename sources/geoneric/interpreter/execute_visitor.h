#pragma once
#include <memory>
#include <stack>
#include "geoneric/core/symbol_table.h"
#include "geoneric/ast/visitor/ast_visitor.h"
#include "geoneric/operation/core/operations.h"


namespace geoneric {

class DataSource;


//! Visitor that executes an abstract syntax tree.
/*!
  This visitor immediately executes expressions as they are encountered during
  the visit.

  <ideas>
  Encapsulate executable stuff in an Executor instance (script, function,
  expression). Each Executor accepts argument values and returns result
  values.

  A script is an anonymous function. The undefined variables are the
  script's arguments, which need to be satisfied by some other means. If this
  doesn't happen, then they are really undefined.

  We need a Context that has a symbol table with currently visible variables
  and their values. Such an instance must be passed into the Executors. That's
  how they get at values. Each Executor can store local variables locally.
  They may update global variables in the Context's symbol table.
  </ideas>

  \sa        .
*/
class ExecuteVisitor:
    public AstVisitor
{

public:

                   ExecuteVisitor      (OperationsPtr const& operations);

                   ~ExecuteVisitor     ();

                   ExecuteVisitor      (ExecuteVisitor&&)=delete;

    ExecuteVisitor& operator=          (ExecuteVisitor&&)=delete;

                   ExecuteVisitor      (ExecuteVisitor const&)=delete;

    ExecuteVisitor& operator=          (ExecuteVisitor const&)=delete;

    void           set_data_source_symbols(
                                        SymbolTable<std::shared_ptr<
                                            DataSource>> const& symbol_table);

    std::stack<std::shared_ptr<Argument>> const&
                   stack               () const;

    void           clear_stack         ();

    SymbolTable<std::shared_ptr<Argument>> const&
                   symbol_table        () const;

private:

    OperationsPtr  _operations;

    //! Stack with values that are passed in and out of expressions.
    std::stack<std::shared_ptr<Argument>> _stack;

    //! Symbol table with values of variables.
    SymbolTable<std::shared_ptr<Argument>> _symbol_table;

    //! Symbol table with data sources for undefined input variables.
    SymbolTable<std::shared_ptr<DataSource>> _data_source_symbol_table;

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (IfVertex& vertex);

    void           Visit               (ModuleVertex& vertex);

    void           Visit               (NameVertex& vertex);

    void           Visit               (NumberVertex<int8_t>& vertex);

    void           Visit               (NumberVertex<int16_t>& vertex);

    void           Visit               (NumberVertex<int32_t>& vertex);

    void           Visit               (NumberVertex<int64_t>& vertex);

    void           Visit               (NumberVertex<uint8_t>& vertex);

    void           Visit               (NumberVertex<uint16_t>& vertex);

    void           Visit               (NumberVertex<uint32_t>& vertex);

    void           Visit               (NumberVertex<uint64_t>& vertex);

    void           Visit               (NumberVertex<float>& vertex);

    void           Visit               (NumberVertex<double>& vertex);

    void           Visit               (OperationVertex& vertex);

    void           Visit               (StringVertex& vertex);

    void           Visit               (SubscriptVertex& vertex);

    void           Visit               (WhileVertex& vertex);

};

} // namespace geoneric
