#include "geoneric/operation/std/int32.h"
#include "geoneric/feature/scalar_attribute.h"
#include "geoneric/operation/core/attribute_argument.h"


namespace geoneric {

Int32::Int32()

    : Operation("int32",
          "Cast the argument to int32 and return the result.",
          {
              Parameter("Input argument",
                  "Argument to cast.",
                  DataTypes::SCALAR | DataTypes::FEATURE,
                  ValueTypes::NUMBER)
          },
          {
              Result("Result value",
                  "Input argument as an int32.",
                  DataTypes::SCALAR | DataTypes::FEATURE,
                  ValueTypes::INT32)
          }
      )

{
}


std::vector<std::shared_ptr<Argument>> Int32::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 1);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    assert(false);

    // AttributeArgument const& attribute_argument(
    //     *std::dynamic_pointer_cast<AttributeArgument>(arguments[0]));
    // Attribute const& attribute(*attribute_argument.attribute());

    // assert(attribute.data_type() == DataType::DT_SCALAR);
    // assert(attribute.value_type() == ValueType::VT_INT64);

    // ScalarAttribute<int64_t> const& value(
    //     dynamic_cast<ScalarAttribute<int64_t> const&>(attribute));

    // int64_t result = std::abs((*value.value())());

    // // // TODO Store result
    // // // TODO Return result

    // return std::vector<std::shared_ptr<Argument>>({
    //     std::shared_ptr<Argument>(new AttributeArgument(
    //         std::shared_ptr<Attribute>(new ScalarAttribute<int64_t>(
    //             std::make_shared<ScalarDomain>(),
    //             std::make_shared<ScalarValue<int64_t>>(result)))))
    // });
}

} // namespace geoneric
