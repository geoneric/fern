#include "geoneric/operation/core/result_type.h"


namespace geoneric {

ResultType::ResultType(
    DataTypes const& data_types,
    ValueTypes const& value_types)

    : _data_types(data_types),
      _value_types(value_types)

{
}


DataTypes ResultType::data_type() const
{
    return _data_types;
}


ValueTypes ResultType::value_type() const
{
    return _value_types;
}



bool ResultType::defined() const
{
    return _data_types.count() > 0 and _value_types.count() > 0;
}


bool ResultType::fixed() const
{
    return _data_types.fixed() && _value_types.fixed();
}


bool ResultType::is_satisfied_by(
    ResultType const& result_type) const
{
    return
        result_type.data_type().is_subset_of(_data_types) &&
        result_type.value_type().is_subset_of(_value_types);
}


bool operator==(
    ResultType const& lhs,
    ResultType const& rhs)
{
    return lhs.data_type() == rhs.data_type() &&
        lhs.value_type() == rhs.value_type();
}


bool operator!=(
    ResultType const& lhs,
    ResultType const& rhs)
{
    return !(lhs == rhs);
}


std::ostream& operator<<(
    std::ostream& stream,
    ResultType const& result_type)
{
    stream << result_type.data_type() << "/" << result_type.value_type();
    return stream;
}

} // namespace geoneric
