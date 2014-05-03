#include "fern/operation/std/read.h"
#include "fern/core/data_name.h"
#include "fern/core/io_error.h"
#include "fern/feature/core/constant_attribute.h"
#include "fern/io/drivers.h"
#include "fern/operation/core/attribute_argument.h"
#include "fern/operation/core/feature_argument.h"


namespace fern {

std::vector<std::shared_ptr<Argument>> read(
        Attribute const& attribute)
{
    ConstantAttribute<String> const& value(
        dynamic_cast<ConstantAttribute<String> const&>(attribute));

    // We have the name of a feature or attribute. We need to read it and
    // return the result.
    // - Assume the name refers to an attribute.
    // - Assume the name refers to a raster that can be read by gdal.

    DataName name(value.values().value());
    std::shared_ptr<Dataset> dataset(
        open_dataset(name.database_pathname().generic_string(),
        OpenMode::READ));
    assert(dataset);

    // Dataset is open, first find out where name.data_pathname is pointing
    // to: feature or attribute. Then request either a feature or an attribute.
    std::vector<std::shared_ptr<Argument>> result;
    if(dataset->contains_feature(name.data_pathname())) {
        result = std::vector<std::shared_ptr<Argument>>({
            std::shared_ptr<Argument>(std::make_shared<FeatureArgument>(
                dataset->read_feature(name.data_pathname())))});
    }
    else if(dataset->contains_attribute(name.data_pathname())) {
        result = std::vector<std::shared_ptr<Argument>>({
            std::shared_ptr<Argument>(std::make_shared<AttributeArgument>(
                dataset->read_attribute(name.data_pathname())))});
    }
    else {
        // TODO Shouldn't this be detected earlier?! Annotate/validate?!
        throw IOError(name.database_pathname().generic_string(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_DATA, name.data_pathname()));
    }

    assert(!result.empty());
    return result;
}


Read::Read()

    : Operation("read",
          "Read a feature or a feature attribute and return the result.",
          {
              Parameter("Feature or attribute name",
                  "Name of feature or attribute to read.",
                  DataTypes::CONSTANT,
                  ValueTypes::STRING)
          },
          {
              Result("Feature read",
                  "Feature read.",
                  ExpressionType(DataTypes::STATIC_FIELD,
                      ValueTypes::NUMBER))
          }
      )

{
}


ExpressionType Read::expression_type(
    size_t index,
    std::vector<ExpressionType> const& /* argument_types */) const
{
    assert(index == 0);

    return results()[0].expression_type();
}


std::vector<std::shared_ptr<Argument>> Read::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 1);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    AttributeArgument const& attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[0]));
    assert(attribute_argument.data_type() == DataType::DT_CONSTANT);
    assert(attribute_argument.value_type() == ValueType::VT_STRING);
    Attribute const& attribute(*attribute_argument.attribute());

    return read(attribute);
}

} // namespace fern
