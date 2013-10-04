#include "geoneric/operation/std/read.h"
#include "geoneric/core/data_name.h"
#include "geoneric/feature/core/constant_attribute.h"
#include "geoneric/io/gdal/drivers.h"
#include "geoneric/io/core/path.h"
#include "geoneric/operation/core/attribute_argument.h"
#include "geoneric/operation/core/feature_argument.h"


namespace geoneric {

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
    std::shared_ptr<Dataset> dataset(open(name.dataset_name()));

    // Dataset is open, first find out where name.data_pathname is pointing
    // to: feature or attribute. Then request either a feature or an attribute.
    std::vector<std::shared_ptr<Argument>> result;
    if(dataset->contains_feature(name.data_pathname())) {
        result = std::vector<std::shared_ptr<Argument>>({
            std::shared_ptr<Argument>(new FeatureArgument(
                dataset->read_feature(name.data_pathname())))});
    }
    else if(dataset->contains_attribute(name.data_pathname())) {
        result = std::vector<std::shared_ptr<Argument>>({
            std::shared_ptr<Argument>(new AttributeArgument(
                dataset->read_attribute(name.data_pathname())))});
    }
    else {
        // TODO raise exception;
        assert(false);
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
                  ResultType(DataTypes::STATIC_FIELD,
                      ValueTypes::NUMBER))
          }
      )

{
}


ResultType Read::result_type(
    size_t index,
    std::vector<ResultType> const& /* argument_types */) const
{
    assert(index == 0);

    return results()[0].result_type();
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

} // namespace geoneric
