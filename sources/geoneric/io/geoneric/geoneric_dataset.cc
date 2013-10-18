#include "geoneric/io/geoneric/geoneric_dataset.h"
#include "geoneric/core/data_name.h"
#include "geoneric/io/geoneric/utils.h"


namespace geoneric {

GeonericDataset::GeonericDataset(
    String const& name,
    OpenMode open_mode)

    : Dataset(name, open_mode),
      _file()

{
    DataName data_name(name);
    _file = open_file(data_name.database_pathname(), open_mode);
}


GeonericDataset::GeonericDataset(
    std::shared_ptr<H5::H5File> const& file,
    String const& name,
    OpenMode open_mode)

    : Dataset(name, open_mode),
      _file(file)

{
}


GeonericDataset::~GeonericDataset()
{
}


size_t GeonericDataset::nr_features() const
{
    return 0u;
}


bool GeonericDataset::contains_feature(
    String const& /* name */) const
{
    return false;
}


bool GeonericDataset::contains_attribute(
    String const& /* name */) const
{
    return false;
}


std::shared_ptr<Feature> GeonericDataset::read_feature(
    String const& /* name */) const
{
    std::shared_ptr<Feature> result;
    return result;
}


std::shared_ptr<Attribute> GeonericDataset::read_attribute(
    String const& /* name */) const
{
    std::shared_ptr<Attribute> result;
    return result;
}


void GeonericDataset::write_attribute(
    Attribute const& /* attribute */,
    String const& /* name */) const
{
}

} // namespace geoneric
