#include "geoneric/interpreter/dataset_source.h"
#include "geoneric/operation/core/attribute_argument.h"
#include "geoneric/io/drivers.h"


namespace geoneric {

DatasetSource::DatasetSource(
    DataName const& data_name)

    : _data_name(data_name),
      _dataset(open_dataset(_data_name.database_pathname(), OpenMode::READ)),
      _expression_type(_dataset->expression_type(_data_name.data_pathname()))

{
}


ExpressionType const& DatasetSource::expression_type() const
{
    return _expression_type;
}


std::shared_ptr<Argument> DatasetSource::read() const
{
    return std::shared_ptr<Argument>(new AttributeArgument(
        _dataset->read_attribute(_data_name.data_pathname())));
}

} // namespace geoneric
