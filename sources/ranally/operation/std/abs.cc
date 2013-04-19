#include "ranally/operation/std/abs.h"
#include "ranally/feature/scalar_attribute.h"
#include "ranally/operation/core/attribute_argument.h"


namespace ranally {

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
}


std::vector<std::shared_ptr<Argument>> Abs::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 1);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    AttributeArgument const& attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[0]));
    Attribute const& attribute(*attribute_argument.attribute());

    assert(attribute.data_type() == DataType::DT_SCALAR);
    assert(attribute.value_type() == ValueType::VT_INT64);

    ScalarAttribute<int64_t> const& value(
        dynamic_cast<ScalarAttribute<int64_t> const&>(attribute));

    int64_t result = std::abs((*value.value())());

    // // TODO Store result
    // // TODO Return result

    return std::vector<std::shared_ptr<Argument>>({
        std::shared_ptr<Argument>(new AttributeArgument(
            std::shared_ptr<Attribute>(new ScalarAttribute<int64_t>(
                std::make_shared<ScalarDomain>(),
                std::make_shared<ScalarValue<int64_t>>(result)))))
    });
}

} // namespace ranally
