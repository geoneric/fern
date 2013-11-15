#include "fern/io/hdf5/hdf5_dataset_driver.h"
#include <cassert>
#include <boost/filesystem.hpp>
#include <cpp/H5Cpp.h>
#include "fern/core/string.h"
#include "fern/io/hdf5/hdf5_dataset.h"


namespace fern {

HDF5DatasetDriver::HDF5DatasetDriver()

    : DatasetDriver()

{
}


HDF5DatasetDriver::~HDF5DatasetDriver()
{
}


bool HDF5DatasetDriver::exists(
    String const& name) const
{
    bool result = false;

    try {
        result = H5::H5File::isHdf5(name.encode_in_utf8().c_str());
    }
    catch(H5::FileIException const&) {
        result = false;
    }

    return result;
}


HDF5Dataset* HDF5DatasetDriver::create(
    String const& name) const
{
    HDF5Dataset* result = 0;

    try {
        unsigned int accessMode = H5F_ACC_TRUNC; // | H5F_ACC_RDWR?
        H5::FileCreatPropList creation_properties =
            H5::FileCreatPropList::DEFAULT;
        H5::FileAccPropList access_properties = H5::FileAccPropList::DEFAULT;
        H5::H5File* file = new H5::H5File(name.encode_in_utf8().c_str(),
            accessMode, creation_properties, access_properties);
        file->flush(H5F_SCOPE_GLOBAL);
        result = new HDF5Dataset(name, file);
    }
    catch(H5::FileIException const& exception) {
        // TODO Raise exception.
        exception.printError(stderr);
        throw std::string("cannot create hdf5 dataset");
    }

    assert(exists(name));
    return result;
}


void HDF5DatasetDriver::remove(
    String const& name) const
{
    if(exists(name)) {
        try {
            boost::filesystem::remove(name.encode_in_utf8().c_str());
        }
        catch(...) {
            // TODO Raise exception.
            throw std::string("cannot remove hdf5 dataset");
        }
    }
}


HDF5Dataset* HDF5DatasetDriver::open(
    String const& name) const
{
    HDF5Dataset* result = 0;

    try {
        H5::H5File* file = new H5::H5File(name.encode_in_utf8().c_str(),
            H5F_ACC_RDONLY);

        result = new HDF5Dataset(name, file);
    }
    catch(H5::FileIException const&) {
        // TODO Raise exception.
        throw std::string("cannot open hdf5 file");
    }

    assert(result);
    return result;
}

} // namespace fern
