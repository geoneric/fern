#pragma once
#include "Ranally/Operation/operation.h"
#include "Ranally/Language/expression_vertex.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OperationVertex:
    public ExpressionVertex
{

    friend class OperationVertexTest;

public:

    virtual        ~OperationVertex    ()=default;

                   OperationVertex     (OperationVertex&&)=delete;

    OperationVertex& operator=         (OperationVertex&&)=delete;

                   OperationVertex     (OperationVertex const&)=delete;

    OperationVertex& operator=         (OperationVertex const&)=delete;

    ExpressionVertices const& expressions() const;

    void           setOperation        (OperationPtr const& operation);

    OperationPtr const& operation      () const;

protected:

                   OperationVertex     (String const& name,
                                        ExpressionVertices const& expressions);

private:

    ExpressionVertices _expressions;

    OperationPtr   _operation;

};

} // namespace ranally
