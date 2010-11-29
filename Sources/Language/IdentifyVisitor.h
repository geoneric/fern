#ifndef INCLUDED_RANALLY_IDENTIFYVISITOR
#define INCLUDED_RANALLY_IDENTIFYVISITOR

#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>

#include "SymbolTable.h"
#include "SyntaxVertex.h"



namespace ranally {

class AssignmentVertex;
class FunctionVertex;
class IfVertex;
class NameVertex;
class OperatorVertex;
class ScriptVertex;
class WhileVertex;

namespace language {

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
class IdentifyVisitor: private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<AssignmentVertex>,
  public Loki::Visitor<FunctionVertex>,
  public Loki::Visitor<IfVertex>,
  public Loki::Visitor<NameVertex>,
  public Loki::Visitor<OperatorVertex>,
  public Loki::Visitor<ScriptVertex>,
  public Loki::Visitor<WhileVertex>
{

  friend class IdentifyVisitorTest;

private:

  SymbolTable      _symbolTable;

  enum Mode {
    Defining,
    Using
  };

  Mode             _mode;

  void             visitStatements     (StatementVertices const& statements);

  void             visitExpressions    (ExpressionVertices const& expressions);

protected:

public:

                   IdentifyVisitor     ();

  /* virtual */    ~IdentifyVisitor    ();

  void             Visit               (AssignmentVertex&);

  void             Visit               (FunctionVertex&);

  void             Visit               (IfVertex&);

  void             Visit               (NameVertex&);

  void             Visit               (OperatorVertex&);

  void             Visit               (ScriptVertex&);

  void             Visit               (WhileVertex&);

  SymbolTable const& symbolTable       () const;

};

} // namespace language
} // namespace ranally

#endif
