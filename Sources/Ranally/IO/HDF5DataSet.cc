#include "Ranally/IO/HDF5DataSet.h"
#include <H5Cpp.h>
#include "Ranally/IO/Feature.h"



namespace ranally {
namespace io {

HDF5DataSet::HDF5DataSet(
  UnicodeString const& name,
  H5::H5File* file)

  : DataSet(name),
    _file(file)

{
  assert(_file);
}



HDF5DataSet::~HDF5DataSet()
{
}



size_t HDF5DataSet::nrFeatures() const
{
  // TODO Find out how many features we have. I think that the number of
  //      features at all levels in the hierarchy must be the same.
  assert(false);
  return 0;
}



Feature* HDF5DataSet::feature(
  size_t /* i */) const
{
  assert(false);
  return 0;
}



void HDF5DataSet::copy(
  DataSet const& dataSet)
{
  for(size_t i = 0; i < dataSet.nrFeatures(); ++i) {
    boost::scoped_ptr<Feature> feature(dataSet.feature(i));
    assert(feature);
    copy(*feature);
  }
}



void HDF5DataSet::copy(
  Feature const& /* feature */)
{
}

} // namespace io
} // namespace ranally

