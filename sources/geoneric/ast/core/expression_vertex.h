#pragma once
#include "geoneric/operation/core/result_types.h"
#include "geoneric/ast/core/statement_vertex.h"


namespace geoneric {

class ExpressionVertex;

typedef std::shared_ptr<ExpressionVertex> ExpressionVertexPtr;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  TODO Make sure expressions support multiple results.

  \sa        .
*/
class ExpressionVertex:
    public StatementVertex
{

    friend class ExpressionVertexTest;

public:

    virtual        ~ExpressionVertex   ()=default;

                   ExpressionVertex    (ExpressionVertex&&)=delete;

    ExpressionVertex& operator=        (ExpressionVertex&&)=delete;

                   ExpressionVertex    (ExpressionVertex const&)=delete;

    ExpressionVertex& operator=        (ExpressionVertex const&)=delete;

    String const&  name                () const;

    void           set_result_types    (ResultTypes const& result_types);

    void           add_result_type     (ResultType const& result_type);

    ResultTypes const& result_types    () const;

    void           set_value           (ExpressionVertexPtr const& value);

    ExpressionVertexPtr const& value   () const;

protected:

                   ExpressionVertex    (String const& name);

                   ExpressionVertex    (int line_nr,
                                        int col_id,
                                        String const& name);

private:

    //! Name of the expression, eg: abs, myDog, 5.
    String         _name;

    ResultTypes    _result_types;

    //! Value of expression, in case this expression is the target of an assignment.
    ExpressionVertexPtr _value;

};

} // namespace geoneric
