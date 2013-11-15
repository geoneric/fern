#include "fern/interpreter/dataset_sync.h"
#include "fern/operation/core/attribute_argument.h"
#include "fern/io/drivers.h"


namespace fern {

DatasetSync::DatasetSync(
    DataName const& data_name)

    : _data_path(data_name.data_pathname()),
      _dataset(open_dataset(data_name.database_pathname(),
          OpenMode::OVERWRITE))

{
}


DatasetSync::DatasetSync(
    std::shared_ptr<Dataset> const& dataset,
    Path const& path)

    : _data_path(path),
      _dataset(dataset)

{
    assert(_dataset);
}


void DatasetSync::write(
    Argument const& argument)
{
    assert(argument.argument_type() == ArgumentType::AT_ATTRIBUTE);
    AttributeArgument const& attribute_argument(
        dynamic_cast<AttributeArgument const&>(argument));
    _dataset->write_attribute(*attribute_argument.attribute(), _data_path);
}

} // namespace fern
