#pragma once
#include <memory>
#include "geoneric/core/type_traits.h"
#include "geoneric/feature/core/constant_attribute.h"
#include "geoneric/operation/core/attribute_argument.h"
#include "geoneric/interpreter/data_source.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class T>
class ConstantSource:
    public DataSource
{

public:

                   ConstantSource      (T const& value);

                   ConstantSource      (ConstantSource const&)=delete;

    ConstantSource& operator=          (ConstantSource const&)=delete;

                   ConstantSource      (ConstantSource&&)=delete;

    ConstantSource& operator=          (ConstantSource&&)=delete;

                   ~ConstantSource     ()=default;

    ExpressionType const& expression_type() const;

    std::shared_ptr<Argument>
                   read                () const;

private:

    T const        _value;

    ExpressionType _expression_type;

};


template<
    class T>
inline ConstantSource<T>::ConstantSource(
    T const& value)

    : DataSource(),
      _value(value),
      _expression_type(DataTypes::CONSTANT, TypeTraits<T>::value_types)

{
}


template<
    class T>
inline ExpressionType const& ConstantSource<T>::expression_type() const
{
    return _expression_type;
}


template<
    class T>
inline std::shared_ptr<Argument> ConstantSource<T>::read() const
{
    return std::shared_ptr<Argument>(new AttributeArgument(
        std::make_shared<ConstantAttribute<T>>(_value)));
}

} // namespace geoneric
