#pragma once
#include "fern/core/data_types.h"
#include "fern/core/value_types.h"


namespace fern {

//! ExpressionType instances keep track of the data and value types of an expression.
/*!
  Once evaluated, an expression results in data with one data type and one
  value type. Before evaluation, it may not be clear yet what the data- and/or
  value type of the expression's result will be. At that time, the
  ExpressionType instance can store multiple data types and value types.

  \sa        ExpressionTypes
*/
class ExpressionType
{

public:

                   ExpressionType      ()=default;

                   ExpressionType      (DataTypes const& data_types,
                                        ValueTypes const& value_types);

                   ExpressionType      (ExpressionType&&)=default;

    ExpressionType& operator=          (ExpressionType&&)=default;

                   ExpressionType      (ExpressionType const&)=default;

    ExpressionType& operator=          (ExpressionType const&)=default;

                   ~ExpressionType     ()=default;

    DataTypes      data_type           () const;

    ValueTypes     value_type          () const;

    bool           defined             () const;

    bool           fixed               () const;

    bool           is_satisfied_by     (
                                  ExpressionType const& expression_type) const;

private:

    DataTypes      _data_types;

    ValueTypes     _value_types;

};


bool               operator==          (ExpressionType const& lhs,
                                        ExpressionType const& rhs);

bool               operator!=          (ExpressionType const& lhs,
                                        ExpressionType const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        ExpressionType const& expression_type);

} // namespace fern
