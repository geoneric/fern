#include "ranally/io/util.h"
#include <memory>
#include "ranally/io/hdf5_dataset_driver.h"


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


std::shared_ptr<Dataset> openDataset(
    String const& /* dataSetName */,
    OpenMode /* openMode */)
{
    std::shared_ptr<Dataset> dataset;

    assert(dataset);
    return dataset;
}

} // Anonymous namespace


void import(
    String const& inputDatasetName,
    String const& outputDatasetName)
{
    std::shared_ptr<Dataset> inputDataset = openDataset(inputDatasetName, Read);
    HDF5DatasetDriver hdf5Driver;
    std::shared_ptr<Dataset> outputDataset(hdf5Driver.create(
        outputDatasetName));
    outputDataset->copy(*inputDataset);
}

} // namespace ranally
