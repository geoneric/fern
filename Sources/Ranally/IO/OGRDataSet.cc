#include "Ranally/IO/OGRDataSet.h"
#include <cassert>
#include "ogrsf_frmts.h"



namespace ranally {
namespace io {

OGRDataSet::OGRDataSet(
  UnicodeString const& name,
  OGRDataSource* dataSource)

  : DataSet(name),
    _dataSource(dataSource)

{
  assert(_dataSource);
}



OGRDataSet::~OGRDataSet()
{
  OGRDataSource::DestroyDataSource(_dataSource);
}

} // namespace io
} // namespace ranally

