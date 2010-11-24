#ifndef INCLUDED_RANALLY_IDENTIFYVISITOR
#define INCLUDED_RANALLY_IDENTIFYVISITOR

#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>

#include "SymbolTable.h"
#include "SyntaxVertex.h"



namespace ranally {

class AssignmentVertex;
class FunctionVertex;
class NameVertex;
class OperatorVertex;
class ScriptVertex;

namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class IdentifyVisitor: private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<AssignmentVertex>,
  public Loki::Visitor<FunctionVertex>,
  public Loki::Visitor<NameVertex>,
  public Loki::Visitor<OperatorVertex>,
  public Loki::Visitor<ScriptVertex>
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

  void             Visit               (NameVertex&);

  void             Visit               (OperatorVertex&);

  void             Visit               (ScriptVertex&);

  SymbolTable const& symbolTable       () const;

};

} // namespace language
} // namespace ranally

#endif
