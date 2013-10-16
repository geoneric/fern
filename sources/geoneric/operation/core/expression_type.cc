#include "geoneric/operation/core/expression_type.h"


namespace geoneric {

//! Constructor.
/*!
  \param     data_types Possible data types the expression may result in.
             This must not be DataTypes::UNKNOWN.
  \param     value_types Possible value types the expression may result in.
             This must not be ValueTypes::UNKNOWN.
*/
ExpressionType::ExpressionType(
    DataTypes const& data_types,
    ValueTypes const& value_types)

    : _data_types(data_types),
      _value_types(value_types)

{
    assert(data_types != DataTypes::UNKNOWN);
    assert(value_types != ValueTypes::UNKNOWN);
}


//! Return the data type of the expression.
/*!
  \return    Data type. If the data type isn't fixed yet, more than one can
             be set.
  \sa        value_type(), fixed()
*/
DataTypes ExpressionType::data_type() const
{
    return _data_types;
}


//! Return the value type of the expression.
/*!
  \return    Value type. If the value type isn't fixed yet, more than one can
             be set.
  \sa        data_type(), fixed()
*/
ValueTypes ExpressionType::value_type() const
{
    return _value_types;
}


//! Return whether at least one data type and one value type is set.
/*!
  \sa        fixed()
*/
bool ExpressionType::defined() const
{
    return _data_types.count() > 0 and _value_types.count() > 0;
}


//! Return whether exactly one data type and one value type is set.
/*!
  \sa        defined()

  This means that the expression's type is known.
*/
bool ExpressionType::fixed() const
{
    return _data_types.fixed() && _value_types.fixed();
}


//! Return whether this instance is satisfied by \a expression_type.
/*!
  An expression type is satisfied by another expression type if the latter is
  a subset of the first one.
*/
bool ExpressionType::is_satisfied_by(
    ExpressionType const& expression_type) const
{
    return
        expression_type.data_type().is_subset_of(_data_types) &&
        expression_type.value_type().is_subset_of(_value_types);
}


bool operator==(
    ExpressionType const& lhs,
    ExpressionType const& rhs)
{
    return lhs.data_type() == rhs.data_type() &&
        lhs.value_type() == rhs.value_type();
}


bool operator!=(
    ExpressionType const& lhs,
    ExpressionType const& rhs)
{
    return !(lhs == rhs);
}


std::ostream& operator<<(
    std::ostream& stream,
    ExpressionType const& expression_type)
{
    stream << expression_type.data_type() << "/"
        << expression_type.value_type();
    return stream;
}

} // namespace geoneric
