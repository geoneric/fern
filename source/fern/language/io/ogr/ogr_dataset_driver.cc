// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/ogr/ogr_dataset_driver.h"
#include <cassert>
#include <ogrsf_frmts.h>


namespace fern {

OGRDatasetDriver::OGRDatasetDriver(
    std::string const& name)

    : _driver(OGRSFDriverRegistrar::GetRegistrar()->GetDriverByName(
          name.encode_in_utf8().c_str()))

{
    assert(_driver);
}


OGRDatasetDriver::~OGRDatasetDriver()
{
}


bool OGRDatasetDriver::exists(
    std::string const& name) const
{
    bool result = false;
    OGRDataSource* data_source = OGRSFDriverRegistrar::Open(
        name.encode_in_utf8().c_str());

    if(data_source) {
        OGRDataSource::DestroyDataSource(data_source);
        result = true;
    }

    return result;
}


OGRDataset* OGRDatasetDriver::create(
    std::string const& name) const
{
    if(exists(name)) {
        remove(name);
    }

    OGRDataSource* data_source = _driver->CreateDataSource(
        name.encode_in_utf8().c_str(), NULL);

    if(!data_source) {
        // TODO exception
        throw std::string("cannot create ogr data source");
    }

    // TODO This doesn't create a (empty) data set on disk. I guess that we
    //      first need to add layers before this works.
    // data_source->SyncToDisk();

    // assert(exists(name));
    return new OGRDataset(name, data_source);
}


void OGRDatasetDriver::remove(
    std::string const& name) const
{
    if(_driver->DeleteDataSource(name.encode_in_utf8().c_str()) !=
        OGRERR_NONE) {
        // TODO Could not remove data source.
        throw std::string("could not remove data source");
    }
}


OGRDataset* OGRDatasetDriver::open(
    std::string const& name) const
{
    // OGRDataSource* data_source = OGRSFDriverRegistrar::Open(
    //   fern::util::encode_in_utf8(name).c_str(), FALSE);
    OGRDataSource* data_source = _driver->Open(
        name.encode_in_utf8().c_str(), FALSE);

    if(!data_source) {
        // TODO exception
        throw std::string("cannot open ogr data source");
    }

    return new OGRDataset(name, data_source);
}

} // namespace fern
