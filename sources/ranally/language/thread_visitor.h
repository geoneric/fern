#pragma once
#include "ranally/language/visitor.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
  \todo      Is it possible to pass pointers as arguments to the Visit functions?
*/
class ThreadVisitor
    : public Visitor
{

    friend class ThreadVisitorTest;

public:

                   ThreadVisitor       ();

                   ~ThreadVisitor      ()=default;

                   ThreadVisitor       (ThreadVisitor&&)=delete;

    ThreadVisitor& operator=           (ThreadVisitor&&)=delete;

                   ThreadVisitor       (ThreadVisitor const&)=delete;

    ThreadVisitor& operator=           (ThreadVisitor const&)=delete;

private:

    //! Last vertex processed on the control flow path.
    SyntaxVertex*  _last_vertex;

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (FunctionVertex& vertex);

    void           Visit               (IfVertex& vertex);

    void           Visit               (NameVertex& vertex);

    template<typename T>
    void           Visit               (NumberVertex<T>& vertex);

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

    void           Visit               (OperatorVertex& vertex);

    void           Visit               (ScriptVertex& vertex);

    void           Visit               (StringVertex& vertex);

    void           Visit               (SyntaxVertex& vertex);

    void           Visit               (WhileVertex& vertex);

};

} // namespace ranally