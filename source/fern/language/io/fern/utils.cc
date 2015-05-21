// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/utils.h"
#include "fern/core/data_name.h"
#include "fern/core/io_error.h"
#include "fern/language/io/core/file.h"


namespace fern {
namespace language {

static std::unique_ptr<HDF5File> open_file_for_overwrite(
    Path const& path)
{
    hid_t id = H5Fcreate(path.native_string().c_str(), H5F_ACC_TRUNC,
        H5P_DEFAULT, H5P_DEFAULT);

    if(id < 0) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_CREATED]);
    }

    H5Fflush(id, H5F_SCOPE_GLOBAL);

    return std::make_unique<HDF5File>(id);
}


static std::unique_ptr<HDF5File> open_file_for_update(
    Path const& path)
{
    if(!file_exists(path)) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::DOES_NOT_EXIST]);
    }

    hid_t id = H5Fopen(path.native_string().c_str(), H5F_ACC_RDWR,
        H5P_DEFAULT);

    if(id < 0) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_WRITTEN]);
    }

    H5Fflush(id, H5F_SCOPE_GLOBAL);

    return std::make_unique<HDF5File>(id);
}


static std::unique_ptr<HDF5File> open_file_for_read(
    Path const& path)
{
    if(!file_exists(path)) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::DOES_NOT_EXIST]);
    }

    hid_t id = H5Fopen(path.native_string().c_str(), H5F_ACC_RDONLY,
        H5P_DEFAULT);

    if(id < 0) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_READ]);
    }

    H5Fflush(id, H5F_SCOPE_GLOBAL);

    return std::make_unique<HDF5File>(id);
}


std::unique_ptr<HDF5File> open_file(
    Path const& path,
    OpenMode open_mode)
{
    std::unique_ptr<HDF5File> result;

    switch(open_mode) {
        case OpenMode::OVERWRITE: {
            result = open_file_for_overwrite(path);
            break;
        }
        case OpenMode::UPDATE: {
            result = open_file_for_update(path);
            break;
        }
        case OpenMode::READ: {
            result = open_file_for_read(path);
            break;
        }
    }

    return result;
}

} // namespace language
} // namespace fern
