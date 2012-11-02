#include "Ranally/IO/Util.h"
#include <memory>
#include "Ranally/IO/HDF5DataSetDriver.h"


namespace ranally {
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


std::shared_ptr<DataSet> openDataSet(
    String const& /* dataSetName */,
    OpenMode /* openMode */)
{
    std::shared_ptr<DataSet> dataset;

    assert(dataset);
    return dataset;
}

} // Anonymous namespace


void import(
    String const& inputDataSetName,
    String const& outputDataSetName)
{
    std::shared_ptr<DataSet> inputDataSet = openDataSet(inputDataSetName, Read);
    HDF5DataSetDriver hdf5Driver;
    std::shared_ptr<DataSet> outputDataSet(hdf5Driver.create(
        outputDataSetName));
    outputDataSet->copy(*inputDataSet);
}

} // namespace ranally
