#include "geoneric/operation/std/read.h"
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

    String const name = value.values().value();
    // TODO Properly parse name. Handle directories, ...
    String const dataset_name = name;
    String const feature_name = Path(name).stem();

    std::shared_ptr<Dataset> dataset(open(dataset_name));

    if(!dataset->contains_feature(feature_name)) {
        // TODO raise exception;
        assert(false);
    }

    std::shared_ptr<Feature> result(dataset->read(feature_name));
    assert(result);

    // We now have a feature containing a box domain and a 2D array value.
    // Just return it. This is what we have, given the argument. Generalize
    // later on.
    return std::vector<std::shared_ptr<Argument>>({
        std::shared_ptr<Argument>(new FeatureArgument(result))
    });
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
              Result("Feature read",
                  "Feature read.",
                  DataTypes::ALL,
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
