#pragma once
#include "fern/core/symbol_table.h"
#include "fern/ast/visitor/ast_visitor.h"


namespace fern {

//! Class for visitors that connect uses of names with their definitions.
/*!
  For example, in the next example, \a a is defined on the first line and used
  on the second.

  \code
  a = 5
  b = a + 3
  \endcode

  This visitor makes sure that the vertex representing the use of \a a is
  connected to the vertex representing the definition of \a a. Also, this
  visitor adds pointers to all uses of names to the corresponding definition
  vertices. This process takes scoping into account.

  \sa        SymbolTable, NameVertex
*/
class IdentifyVisitor:
    public AstVisitor
{

    friend class IdentifyVisitorTest;

public:

                   IdentifyVisitor     ();

                   ~IdentifyVisitor    ()=default;

                   IdentifyVisitor     (IdentifyVisitor&&)=delete;

    IdentifyVisitor& operator=         (IdentifyVisitor&&)=delete;

                   IdentifyVisitor     (IdentifyVisitor const&)=delete;

    IdentifyVisitor& operator=         (IdentifyVisitor const&)=delete;

    SymbolTable<NameVertex*> const& symbol_table() const;

private:

    enum Mode {
        Defining,
        Using
    };

    SymbolTable<NameVertex*> _symbol_table;

    Mode           _mode;

    void           Visit               (AssignmentVertex&);

    void           Visit               (FunctionCallVertex&);

    void           Visit               (IfVertex&);

    void           Visit               (NameVertex&);

    void           Visit               (OperatorVertex&);

    void           Visit               (ModuleVertex&);

    // void           Visit               (SubscriptVertex&);

    void           Visit               (WhileVertex&);

};

} // namespace fern
