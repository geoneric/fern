#ifndef INCLUDED_RANALLY_LANGUAGE_OPTIMIZEVISITOR
#define INCLUDED_RANALLY_LANGUAGE_OPTIMIZEVISITOR

#include "Ranally/Language/Visitor.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OptimizeVisitor
  : public Visitor
{

  friend class OptimizeVisitorTest;

public:

                   OptimizeVisitor     ();

                   ~OptimizeVisitor    ();

private:

  void             Visit               (NameVertex& vertex);

};

} // namespace language
} // namespace ranally

#endif
