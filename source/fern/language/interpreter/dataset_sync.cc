// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/interpreter/dataset_sync.h"
#include "fern/language/operation/core/attribute_argument.h"
#include "fern/language/io/drivers.h"


namespace fern {
namespace language {

DatasetSync::DatasetSync(
    DataName const& data_name)

    : _data_path(data_name.data_pathname()),
      _dataset(open_dataset(data_name.database_pathname().generic_string(),
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


std::shared_ptr<Dataset> const& DatasetSync::dataset() const
{
    assert(_dataset);
    return _dataset;
}

} // namespace language
} // namespace fern
