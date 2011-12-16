#ifndef INCLUDED_RANALLY_LANGUAGE_EXECUTEVISITOR
#define INCLUDED_RANALLY_LANGUAGE_EXECUTEVISITOR

#include "Ranally/Language/Visitor.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ExecuteVisitor:
  public Visitor
{

  friend class ExecuteVisitorTest;

public:

                   ExecuteVisitor      ();

                   ~ExecuteVisitor     ();

private:

};

} // namespace language
} // namespace ranally

#endif
