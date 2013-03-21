#pragma once
#include "ranally/operation/result_types.h"
#include "ranally/language/statement_vertex.h"


namespace ranally {

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

    // void           set_data_type    (operation::DataType data_type);

    // operation::DataType data_type         () const;

    // void           set_value_type   (operation::ValueType value_type);

    // operation::ValueType value_type       () const;

    void           set_result_types    (ResultTypes const& result_types);

    void           add_result_type     (ResultType const& result_type);

    ResultTypes const& result_types    () const;

    DataTypes      data_type           (size_t index) const;

    ValueTypes     value_type          (size_t index) const;

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

    // //! Data type of the result(s) of the expression.
    // operation::DataType _data_type;

    // //! Value type of the result(s) of the expression.
    // operation::ValueType _value_type;

    // typedef boost::tuple<operation::DataType, operation::ValueType> ResultType;

    ResultTypes    _result_types;

    //! Value of expression, in case this expression is the target of an assignment.
    ExpressionVertexPtr _value;

};

} // namespace ranally
