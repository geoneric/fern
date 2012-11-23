#include "Ranally/IO/ogr_dataset_driver.h"
#include <cassert>
#include "ogrsf_frmts.h"
#include "Ranally/Util/string.h"


namespace ranally {

OGRDatasetDriver::OGRDatasetDriver(
    String const& name)

    : _driver(OGRSFDriverRegistrar::GetRegistrar()->GetDriverByName(
          name.encodeInUTF8().c_str()))

{
    assert(_driver);
}


OGRDatasetDriver::~OGRDatasetDriver()
{
}


bool OGRDatasetDriver::exists(
    String const& name) const
{
    bool result = false;
    OGRDataSource* dataSource = OGRSFDriverRegistrar::Open(
        name.encodeInUTF8().c_str());

    if(dataSource) {
        OGRDataSource::DestroyDataSource(dataSource);
        result = true;
    }

    return result;
}


OGRDataset* OGRDatasetDriver::create(
    String const& name) const
{
    if(exists(name)) {
        remove(name);
    }

    OGRDataSource* dataSource = _driver->CreateDataSource(
        name.encodeInUTF8().c_str(), NULL);

    if(!dataSource) {
        // TODO exception
        throw std::string("cannot create ogr data source");
    }

    // TODO This doesn't create a (empty) data set on disk. I guess that we
    //      first need to add layers before this works.
    // dataSource->SyncToDisk();

    // assert(exists(name));
    return new OGRDataset(name, dataSource);
}


void OGRDatasetDriver::remove(
    String const& name) const
{
    if(_driver->DeleteDataSource(name.encodeInUTF8().c_str()) !=
        OGRERR_NONE) {
        // TODO Could not remove data source.
        throw std::string("could not remove data source");
    }
}


OGRDataset* OGRDatasetDriver::open(
    String const& name) const
{
    // OGRDataSource* dataSource = OGRSFDriverRegistrar::Open(
    //   ranally::util::encodeInUTF8(name).c_str(), FALSE);
    OGRDataSource* dataSource = _driver->Open(
        name.encodeInUTF8().c_str(), FALSE);

    if(!dataSource) {
        // TODO exception
        throw std::string("cannot open ogr data source");
    }

    return new OGRDataset(name, dataSource);
}

} // namespace ranally
