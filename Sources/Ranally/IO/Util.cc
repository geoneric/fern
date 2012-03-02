#include "Ranally/IO/Util.h"
#include <boost/shared_ptr.hpp>
#include "Ranally/IO/HDF5DataSetDriver.h"



namespace ranally {
namespace io {
namespace {

//! Opening modes for data sets.
enum OpenMode {
  //! Open data set for reading.
  Read,

  //! Open data set for writing. This will truncate the file.
  Write,

  //! Open data set for writing, updating existing information.
  Update
};



boost::shared_ptr<DataSet> openDataSet(
  UnicodeString const& /* dataSetName */,
  OpenMode /* openMode */)
{
  boost::shared_ptr<DataSet> dataset;

  assert(dataset);
  return dataset;
}

} // Anonymous namespace



void import(
  UnicodeString const& inputDataSetName,
  UnicodeString const& outputDataSetName)
{
  boost::shared_ptr<DataSet> inputDataSet = openDataSet(inputDataSetName, Read);
  HDF5DataSetDriver hdf5Driver;
  boost::shared_ptr<DataSet> outputDataSet(hdf5Driver.create(
    outputDataSetName));
  outputDataSet->copy(*inputDataSet);
}

} // namespace io
} // namespace ranally

