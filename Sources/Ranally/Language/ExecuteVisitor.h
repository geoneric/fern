#pragma once
#include "Ranally/Language/Visitor.h"


namespace ranally {

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

                   ExecuteVisitor      ()=default;

                   ~ExecuteVisitor     ()=default;

                   ExecuteVisitor      (ExecuteVisitor&&)=delete;

    ExecuteVisitor& operator=          (ExecuteVisitor&&)=delete;

                   ExecuteVisitor      (ExecuteVisitor const&)=delete;

    ExecuteVisitor& operator=          (ExecuteVisitor const&)=delete;

private:

    void           Visit               (OperationVertex& vertex);

};

} // namespace ranally
