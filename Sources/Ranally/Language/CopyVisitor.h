#ifndef INCLUDED_RANALLY_LANGUAGE_COPYVISITOR
#define INCLUDED_RANALLY_LANGUAGE_COPYVISITOR

#include "Ranally/Language/Visitor.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class CopyVisitor:
  public Visitor
{

  friend class CopyVisitorTest;

public:

                   CopyVisitor         ();

                   ~CopyVisitor        ();

  StatementVertices const& statements  () const;

protected:

private:

  StatementVertices _statements;

  void             Visit               (AssignmentVertex&);

  void             Visit               (FunctionVertex&);

  void             Visit               (IfVertex&);

  void             Visit               (NameVertex&);

  void             Visit               (OperatorVertex&);

  void             Visit               (ScriptVertex&);

  void             Visit               (WhileVertex&);

};

} // namespace language
} // namespace ranally

#endif
