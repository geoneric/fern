#include "ranally/operation/std/abs.h"
#include "ranally/feature/scalar_attribute.h"
#include "ranally/operation/core/attribute_argument.h"


namespace ranally {

template<
    typename T>
std::vector<std::shared_ptr<Argument>> abs(
        Attribute const& attribute)
{
    ScalarAttribute<T> const& value(
        dynamic_cast<ScalarAttribute<T> const&>(attribute));
    T result = std::abs((*value.value())());

    return std::vector<std::shared_ptr<Argument>>({
        std::shared_ptr<Argument>(new AttributeArgument(
            std::shared_ptr<Attribute>(new ScalarAttribute<T>(
                std::make_shared<ScalarDomain>(),
                std::make_shared<ScalarValue<T>>(result)))))
    });
}


Abs::Abs()

    : Operation("abs",
          "Calculate the absolute value of the argument and return the result.",
          {
              Parameter("Input argument",
                  "Argument to calculate absolute value of.",
                  DataTypes::SCALAR | DataTypes::FEATURE,
                  ValueTypes::NUMBER)
          },
          {
              Result("Result value",
                  "Absolute value of the input argument.",
                  DataTypes::SCALAR | DataTypes::FEATURE,
                  ValueTypes::NUMBER)
          }
      )

{
    std::cout << parameters()[0].data_types() << std::endl;
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

    switch(attribute.data_type()) {
        case DataType::DT_SCALAR: {
            switch(attribute.value_type()) {
                case ValueType::VT_UINT8: {
                    result = abs<uint8_t>(attribute);
                    break;
                }
                case ValueType::VT_INT8: {
                    result = abs<int8_t>(attribute);
                    break;
                }
                case ValueType::VT_UINT16: {
                    result = abs<uint16_t>(attribute);
                    break;
                }
                case ValueType::VT_INT16: {
                    result = abs<int16_t>(attribute);
                    break;
                }
                case ValueType::VT_UINT32: {
                    result = abs<uint32_t>(attribute);
                    break;
                }
                case ValueType::VT_INT32: {
                    result = abs<int32_t>(attribute);
                    break;
                }
                case ValueType::VT_UINT64: {
                    result = abs<uint64_t>(attribute);
                    break;
                }
                case ValueType::VT_INT64: {
                    result = abs<int64_t>(attribute);
                    break;
                }
                case ValueType::VT_FLOAT32: {
                    result = abs<float>(attribute);
                    break;
                }
                case ValueType::VT_FLOAT64: {
                    result = abs<double>(attribute);
                    break;
                }
                case ValueType::VT_STRING: {
                    assert(false);
                    break;
                }
            }

            break;
        }
        case DataType::DT_POINT: {
            assert(false);
            break;
        }
        case DataType::DT_LINE: {
            assert(false);
            break;
        }
        case DataType::DT_POLYGON: {
            assert(false);
            break;
        }
    }

    return result;
}

} // namespace ranally
