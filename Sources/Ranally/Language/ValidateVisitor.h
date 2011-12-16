#ifndef INCLUDED_RANALLY_LANGUAGE_VALIDATEVISITOR
#define INCLUDED_RANALLY_LANGUAGE_VALIDATEVISITOR

#include "Ranally/Operation/Operations.h"
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

                   ValidateVisitor     (
                                  operation::OperationsPtr const& operations);

                   ~ValidateVisitor    ();

private:

  operation::OperationsPtr _operations;

  void             Visit               (FunctionVertex& vertex);

};

} // namespace language
} // namespace ranally

#endif
