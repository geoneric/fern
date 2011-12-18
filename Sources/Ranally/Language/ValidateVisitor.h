#ifndef INCLUDED_RANALLY_LANGUAGE_VALIDATEVISITOR
#define INCLUDED_RANALLY_LANGUAGE_VALIDATEVISITOR

#include "Ranally/Language/Visitor.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ValidateVisitor:
  public Visitor
{

  friend class ValidateVisitorTest;

public:

                   ValidateVisitor     ();

                   ~ValidateVisitor    ();

private:

  void             Visit               (FunctionVertex& vertex);

};

} // namespace language
} // namespace ranally

#endif
