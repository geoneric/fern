#pragma once
#include "ranally/language/visitor.h"


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

    void           Visit               (NumberVertex<int8_t>& vertex);

    void           Visit               (NumberVertex<int16_t>& vertex);

    void           Visit               (NumberVertex<int32_t>& vertex);

    void           Visit               (NumberVertex<int64_t>& vertex);

    void           Visit               (NumberVertex<uint8_t>& vertex);

    void           Visit               (NumberVertex<uint16_t>& vertex);

    void           Visit               (NumberVertex<uint32_t>& vertex);

    void           Visit               (NumberVertex<uint64_t>& vertex);

    void           Visit               (NumberVertex<float>& vertex);

    void           Visit               (NumberVertex<double>& vertex);

    void           Visit               (OperationVertex& vertex);

};

} // namespace ranally
