#pragma once
#include "ranally/core/stack.h"
#include "ranally/language/visitor.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  Encapsulate executable stuff in an Executor instance (script, function,
  expression). Each Executor accepts argument values and returns result
  values.

  A script is an anonymous function. The undefined variables are the
  script's arguments, which need to satisfied by some other means. If this
  doesn't happen, then they are really undefined.

  We need a Context that has a symbol table with currently visible variables
  and their values. Such an instance must be passed into the Executors. That's
  how they get at values. Each Executor can store local variables locally.
  They may update global variables in the Context's symbol table.

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

    //! Stack with values that are passed in and out of expressions.
    Stack          _stack;

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (IfVertex& vertex);

    void           Visit               (NameVertex& vertex);

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

    void           Visit               (StringVertex& vertex);

    void           Visit               (SubscriptVertex& vertex);

    void           Visit               (WhileVertex& vertex);

};

} // namespace ranally
