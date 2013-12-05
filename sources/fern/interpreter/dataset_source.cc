#include "fern/interpreter/dataset_source.h"
#include "fern/operation/core/attribute_argument.h"
#include "fern/io/drivers.h"


namespace fern {

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
#ifndef NDEBUG
    set_data_has_been_read();
#endif
    return std::shared_ptr<Argument>(std::make_shared<AttributeArgument>(
        _dataset->read_attribute(_data_name.data_pathname())));
}

} // namespace fern
