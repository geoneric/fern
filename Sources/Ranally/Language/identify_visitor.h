#pragma once
#include "Ranally/Language/symbol_table.h"
#include "Ranally/Language/visitor.h"


namespace ranally {

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
    public Visitor
{

    friend class IdentifyVisitorTest;

public:

                   IdentifyVisitor     ();

                   ~IdentifyVisitor    ()=default;

                   IdentifyVisitor     (IdentifyVisitor&&)=delete;

    IdentifyVisitor& operator=         (IdentifyVisitor&&)=delete;

                   IdentifyVisitor     (IdentifyVisitor const&)=delete;

    IdentifyVisitor& operator=         (IdentifyVisitor const&)=delete;

    SymbolTable const& symbolTable     () const;

private:

    enum Mode {
        Defining,
        Using
    };

    SymbolTable    _symbolTable;

    Mode           _mode;

    void           Visit               (AssignmentVertex&);

    void           Visit               (FunctionVertex&);

    void           Visit               (IfVertex&);

    void           Visit               (NameVertex&);

    void           Visit               (OperatorVertex&);

    void           Visit               (ScriptVertex&);

    void           Visit               (WhileVertex&);

};

} // namespace ranally
