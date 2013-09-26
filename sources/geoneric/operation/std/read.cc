#include "geoneric/operation/std/read.h"
#include "geoneric/operation/core/attribute_argument.h"


namespace geoneric {

std::vector<std::shared_ptr<Argument>> read(
        Attribute const& /* attribute */)
{
    // ScalarAttribute<String> const& value(
    //     dynamic_cast<ScalarAttribute<String> const&>(attribute));
    // TODO: T result = std::read((*value.value())());

    // return std::vector<std::shared_ptr<Argument>>({
    //     std::shared_ptr<Argument>(new AttributeArgument(
    //         std::shared_ptr<Attribute>(new ScalarAttribute<T>(
    //             std::make_shared<ScalarDomain>(),
    //             std::make_shared<ScalarValue<T>>(result)))))
    // });

    return std::vector<std::shared_ptr<Argument>>();
}


Read::Read()

    : Operation("read",
          "Read a feature or a feature attribute and return the result.",
          {
              Parameter("Feature or attribute name",
                  "Name of feature or attribute to read.",
                  DataTypes::SCALAR,
                  ValueTypes::STRING)
          },
          {
              // TODO: In case a feature is read, what should the data type be?
              Result("Feature read",
                  "Feature read.",
                  DataTypes::FEATURE,
                  ValueTypes::ALL)
          }
      )

{
}


std::vector<std::shared_ptr<Argument>> Read::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 1);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    AttributeArgument const& attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[0]));
    assert(attribute_argument.data_type() == DataType::DT_SCALAR);
    assert(attribute_argument.value_type() == ValueType::VT_STRING);
    Attribute const& attribute(*attribute_argument.attribute());

    return read(attribute);
}

} // namespace geoneric
