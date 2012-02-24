#include "Ranally/IO/OGRDataSetDriver.h"
#include <cassert>
#include "ogrsf_frmts.h"
#include "Ranally/Util/String.h"



namespace ranally {
namespace io {

OGRDataSetDriver::OGRDataSetDriver()
{
}



OGRDataSetDriver::~OGRDataSetDriver()
{
}



bool OGRDataSetDriver::exists(
  UnicodeString const& name) const
{
  bool result = false;
  OGRDataSource* dataSource = OGRSFDriverRegistrar::Open(
    ranally::util::encodeInUTF8(name).c_str());

  if(dataSource) {
    OGRDataSource::DestroyDataSource(dataSource);
    result = true;
  }

  return result;
}



OGRDataSet* OGRDataSetDriver::create(
  UnicodeString const& name) const
{
  std::string driverName = "ESRI Shapefile";
  OGRSFDriver* driver = OGRSFDriverRegistrar::GetRegistrar()->GetDriverByName(
    driverName.c_str());
  assert(driver);

  OGRDataSource* dataSource = driver->CreateDataSource(
    ranally::util::encodeInUTF8(name).c_str(), NULL);

  if(!dataSource) {
    // TODO exception
    throw std::string("cannot create ogr data source");
  }

  // TODO This doesn't create a (empty) data set on disk. I guess that we
  //      first need to add layers before this works.
  // dataSource->SyncToDisk();

  // assert(exists(name));
  return new OGRDataSet(name, dataSource);
}



void OGRDataSetDriver::remove(
  UnicodeString const& /* name */) const
{
}



OGRDataSet* OGRDataSetDriver::open(
  UnicodeString const& name) const
{
  OGRDataSource* dataSource = OGRSFDriverRegistrar::Open(
    ranally::util::encodeInUTF8(name).c_str(), FALSE);

  if(!dataSource) {
    // TODO exception
    throw std::string("cannot open ogr data source");
  }

  return new OGRDataSet(name, dataSource);
}

} // namespace io
} // namespace ranally

