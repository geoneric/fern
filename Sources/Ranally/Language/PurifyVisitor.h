#ifndef INCLUDED_RANALLY_LANGUAGE_PURIFYVISITOR
#define INCLUDED_RANALLY_LANGUAGE_PURIFYVISITOR

#include "Ranally/Language/Visitor.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PurifyVisitor
  : public Visitor
{

  friend class PurifyVisitorTest;

public:

                   PurifyVisitor       ();

                   ~PurifyVisitor      ();

private:

};

} // namespace language
} // namespace ranally

#endif
