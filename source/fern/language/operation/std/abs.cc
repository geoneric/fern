// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/operation/std/abs.h"
#include "fern/core/value_type_traits.h"
#include "fern/language/feature/core/attributes.h"
#include "fern/language/operation/core/attribute_argument.h"
#include "fern/language/operation/core/expression_type_calculation.h"


namespace fern {

template<
    typename T>
std::vector<std::shared_ptr<Argument>> abs(
        ConstantAttribute<T> const& attribute)
{
    T result = std::abs(attribute.values().value());

    return std::vector<std::shared_ptr<Argument>>({
        std::shared_ptr<Argument>(std::make_shared<AttributeArgument>(
            std::make_shared<ConstantAttribute<T>>(result)))
    });
}


template<
    typename T>
std::vector<std::shared_ptr<Argument>> abs(
        FieldAttribute<T> const& attribute)
{
    FieldAttributePtr<T> result(std::make_shared<FieldAttribute<T>>());

    for(auto const& pair: attribute.domain()) {
        FieldValue<T> const& source_array(*attribute.values().value(
            pair.first));
        assert(source_array.size() > 0);
        FieldValuePtr<T> destination_array_ptr(std::make_shared<FieldValue<T>>(
            extents[source_array.size()][source_array[0].size()]));
        FieldValue<T>& destination_array(*destination_array_ptr);

        destination_array.mask() = source_array.mask();

        for(size_t row = 0; row < source_array.shape()[0]; ++row) {
            for(size_t col = 0; col < source_array.shape()[1]; ++col) {
                if(!source_array.mask()[row][col]) {
                    destination_array[row][col] = std::abs(
                        source_array[row][col]);
                }
            }
        }

        // TODO perform abs...
        // - abs with a single value.
        // - abs with two arrays, without a mask
        // - ...
        // - ...

        // TODO share or copy domain...

        result->add(pair.second, destination_array_ptr);
    }

    return std::vector<std::shared_ptr<Argument>>({
        std::shared_ptr<Argument>(std::make_shared<AttributeArgument>(
            result))
    });
}


Abs::Abs()

    : Operation("abs",
          "Calculate the absolute value of the argument and return the result.",
          {
              Parameter("Input argument",
                  "Argument to calculate absolute value of.",
                  DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                  ValueTypes::NUMBER)
          },
          {
              Result("Result value",
                  "Absolute value of the input argument.",
                  ExpressionType(DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                      ValueTypes::NUMBER))
          }
      )

{
}


ExpressionType Abs::expression_type(
    size_t index,
    std::vector<ExpressionType> const& argument_types) const
{
    return standard_expression_type(*this, index, argument_types);
}


#define CONSTANT_CASE(                                                         \
        value_type)                                                            \
    case value_type: {                                                         \
        using type = ValueTypeTraits<value_type>::type;                        \
        result = abs<type>(dynamic_cast<ConstantAttribute<type> const&>(       \
            attribute));                                                       \
        break;                                                                 \
    }

#define FIELD_CASE(                                                            \
        value_type)                                                            \
    case value_type: {                                                         \
        using type = ValueTypeTraits<value_type>::type;                        \
        result = abs<type>(dynamic_cast<FieldAttribute<type> const&>(          \
            attribute));                                                       \
        break;                                                                 \
    }


std::vector<std::shared_ptr<Argument>> Abs::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 1);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    AttributeArgument const& attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[0]));
    Attribute const& attribute(*attribute_argument.attribute());
    std::vector<std::shared_ptr<Argument>> result;

    switch(attribute_argument.data_type()) {
        case DataType::DT_CONSTANT: {
            switch(attribute_argument.value_type()) {
                CONSTANT_CASE(ValueType::VT_UINT8)
                CONSTANT_CASE(ValueType::VT_INT8)
                CONSTANT_CASE(ValueType::VT_UINT16)
                CONSTANT_CASE(ValueType::VT_INT16)
                CONSTANT_CASE(ValueType::VT_UINT32)
                CONSTANT_CASE(ValueType::VT_INT32)
                CONSTANT_CASE(ValueType::VT_UINT64)
                CONSTANT_CASE(ValueType::VT_INT64)
                CONSTANT_CASE(ValueType::VT_FLOAT32)
                CONSTANT_CASE(ValueType::VT_FLOAT64)
                case ValueType::VT_BOOL:
                case ValueType::VT_CHAR:
                case ValueType::VT_STRING: {
                    assert(false);
                    break;
                }
            }
            break;
        }
        case DataType::DT_STATIC_FIELD: {
            switch(attribute_argument.value_type()) {
                FIELD_CASE(ValueType::VT_UINT8)
                FIELD_CASE(ValueType::VT_INT8)
                FIELD_CASE(ValueType::VT_UINT16)
                FIELD_CASE(ValueType::VT_INT16)
                FIELD_CASE(ValueType::VT_UINT32)
                FIELD_CASE(ValueType::VT_INT32)
                FIELD_CASE(ValueType::VT_UINT64)
                FIELD_CASE(ValueType::VT_INT64)
                FIELD_CASE(ValueType::VT_FLOAT32)
                FIELD_CASE(ValueType::VT_FLOAT64)
                case ValueType::VT_BOOL:
                case ValueType::VT_CHAR:
                case ValueType::VT_STRING: {
                    assert(false);
                    break;
                }
            }
            break;
        }
        // case DataType::DT_POINT: {
        //     assert(false);
        //     break;
        // }
        // case DataType::DT_LINE: {
        //     assert(false);
        //     break;
        // }
        // case DataType::DT_POLYGON: {
        //     assert(false);
        //     break;
        // }
    }

    return result;
}

} // namespace fern
