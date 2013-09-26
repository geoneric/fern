#include "geoneric/operation/std/add.h"
#include "geoneric/operation/core/attribute_argument.h"


namespace geoneric {

Add::Add()

    : Operation("add",
          "Add the two argument values and return the result.",
          {
              Parameter("First input argument",
                  "First argument to add.",
                  DataTypes::SCALAR | DataTypes::FEATURE,
                  ValueTypes::NUMBER),
              Parameter("Second input argument",
                  "Second argument to add.",
                  DataTypes::SCALAR | DataTypes::FEATURE,
                  ValueTypes::NUMBER)
          },
          {
              Result("Result value",
                  "Input arguments, added together.",
                  DataTypes::SCALAR | DataTypes::FEATURE,
                  ValueTypes::NUMBER)
          }
      )

{
}


std::vector<std::shared_ptr<Argument>> Add::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 2);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);
    assert(arguments[1]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    AttributeArgument const& attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[0]));
    assert(attribute_argument.data_type() == DataType::DT_SCALAR);
    assert(attribute_argument.value_type() == ValueType::VT_INT64);

    // Attribute const& attribute(*attribute_argument.attribute());

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

    return std::vector<std::shared_ptr<Argument>>();
}

} // namespace geoneric
