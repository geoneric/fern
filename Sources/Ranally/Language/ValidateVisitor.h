#pragma once
#include "Ranally/Language/Visitor.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  This visitor can assume the tree is fully annotated. Any missing
  information must be reported. It means that the information is not
  available. The AnnotateVisitor tries its best to find information but
  won't report errors. That's the task of the ValidateVisitor.

  \sa        AnnotateVisitor
*/
class ValidateVisitor:
    public Visitor
{

    friend class ValidateVisitorTest;

public:

                   ValidateVisitor     ();

                   ~ValidateVisitor    ();

private:

    void           Visit               (NameVertex& vertex);

    void           Visit               (FunctionVertex& vertex);

};

} // namespace ranally
