#include "geoneric/operation/std/write.h"
#include "geoneric/core/data_name.h"
#include "geoneric/core/value_type_traits.h"
#include "geoneric/feature/core/attributes.h"
#include "geoneric/io/drivers.h"
#include "geoneric/operation/core/attribute_argument.h"


namespace geoneric {

std::vector<std::shared_ptr<Argument>> write(
        Attribute const& data_attribute,
        Attribute const& name_attribute,
        Attribute const& format_attribute)
{
    String const& name(dynamic_cast<ConstantAttribute<String> const&>(
        name_attribute).values().value());
    String const& format_name(dynamic_cast<ConstantAttribute<String> const&>(
        format_attribute).values().value());

    DataName data_name(name);
    std::shared_ptr<Dataset> dataset;

    if(!dataset_exists(data_name.database_pathname(), OpenMode::UPDATE,
            format_name)) {
        dataset = create_dataset(data_attribute, data_name.database_pathname(),
            format_name);
    }
    else {
        dataset = open_dataset(data_name.database_pathname(), OpenMode::UPDATE,
            format_name);
    }

    String attribute_name = data_name.data_pathname();

    // if(attribute_name == "/") {
    //     // Short hand notation is used for the attribute name.
    //     attribute_name = Path(data_name.database_pathname()).stem();
    // }

    assert(dataset);
    dataset->write_attribute(data_attribute, attribute_name);

    return std::vector<std::shared_ptr<Argument>>();
}


Write::Write()

    : Operation("write",
          "Write a feature or a feature attribute and return the result.",
          {
              Parameter("Feature or attribute",
                  "Feature or attribute to write.",
                  DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                  ValueTypes::NUMBER),
              Parameter("Name",
                  "Name of feature or attribute to write.",
                  DataTypes::CONSTANT,
                  ValueTypes::STRING),
              Parameter("Format",
                  "Name of format of dataset to write.",
                  DataTypes::CONSTANT,
                  ValueTypes::STRING)
          },
          {
          }
      )

{
}


ExpressionType Write::expression_type(
    size_t index,
    std::vector<ExpressionType> const& /* argument_types */) const
{
    assert(index == 0);

    return ExpressionType();
}


std::vector<std::shared_ptr<Argument>> Write::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 3);

    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);
    AttributeArgument const& data_attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[0]));

    assert(arguments[1]->argument_type() == ArgumentType::AT_ATTRIBUTE);
    AttributeArgument const& name_attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[1]));
    assert(name_attribute_argument.data_type() == DataType::DT_CONSTANT);
    assert(name_attribute_argument.value_type() == ValueType::VT_STRING);

    assert(arguments[2]->argument_type() == ArgumentType::AT_ATTRIBUTE);
    AttributeArgument const& format_attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[2]));
    assert(format_attribute_argument.data_type() == DataType::DT_CONSTANT);
    assert(format_attribute_argument.value_type() == ValueType::VT_STRING);

    return write(
        *data_attribute_argument.attribute(),
        *name_attribute_argument.attribute(),
        *format_attribute_argument.attribute());
}

} // namespace geoneric
