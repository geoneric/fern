#include "fern/io/core/util.h"
#include <memory>
// #include "fern/io/hdf5_dataset_driver.h"


namespace fern {
namespace {

// //! Opening modes for data sets.
// enum OpenMode {
//     //! Open data set for reading.
//     Read,
// 
//     //! Open data set for writing. This will truncate the file.
//     Write,
// 
//     //! Open data set for writing, updating existing information.
//     Update
// };
// 
// 
// std::shared_ptr<Dataset> open_dataset(
//     String const& /* dataset_name */,
//     OpenMode /* open_mode */)
// {
//     std::shared_ptr<Dataset> dataset;
// 
//     assert(dataset);
//     return dataset;
// }

} // Anonymous namespace


// void import(
//     String const& input_dataset_name,
//     String const& output_dataset_name)
// {
//     std::shared_ptr<Dataset> input_dataset = open_dataset(input_dataset_name,
//         Read);
//     HDF5DatasetDriver hdf5_driver;
//     std::shared_ptr<Dataset> output_dataset(hdf5_driver.create(
//         output_dataset_name));
//     output_dataset->copy(*input_dataset);
// }

} // namespace fern
