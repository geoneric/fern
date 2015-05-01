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

static std::shared_ptr<H5::H5File> open_file_for_overwrite(
    Path const& path)
{
    std::shared_ptr<H5::H5File> result;

    try {
        unsigned int access_mode = H5F_ACC_TRUNC;
        H5::FileCreatPropList creation_properties =
            H5::FileCreatPropList::DEFAULT;
        H5::FileAccPropList access_properties = H5::FileAccPropList::DEFAULT;
        result = std::make_shared<H5::H5File>(path.native_string().c_str(),
            access_mode, creation_properties, access_properties);
        result->flush(H5F_SCOPE_GLOBAL);
    }
    catch(H5::FileIException const& /* exception */) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_CREATED]);
    }

    return result;
}


static std::shared_ptr<H5::H5File> open_file_for_update(
    Path const& path)
{
    std::shared_ptr<H5::H5File> result;

    if(!file_exists(path)) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::DOES_NOT_EXIST]);
    }

    try {
        unsigned int access_mode = H5F_ACC_RDWR;
        H5::FileCreatPropList creation_properties =
            H5::FileCreatPropList::DEFAULT;
        H5::FileAccPropList access_properties = H5::FileAccPropList::DEFAULT;
        result = std::make_shared<H5::H5File>(
            path.native_string().c_str(), access_mode, creation_properties,
            access_properties);
        result->flush(H5F_SCOPE_GLOBAL);
    }
    catch(H5::FileIException const& /* exception */) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_WRITTEN]);
    }

    return result;
}


static std::shared_ptr<H5::H5File> open_file_for_read(
    Path const& path)
{
    std::shared_ptr<H5::H5File> result;

    if(!file_exists(path)) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::DOES_NOT_EXIST]);
    }

    try {
        unsigned int access_mode = H5F_ACC_RDONLY;
        H5::FileCreatPropList creation_properties =
            H5::FileCreatPropList::DEFAULT;
        H5::FileAccPropList access_properties = H5::FileAccPropList::DEFAULT;
        result = std::make_shared<H5::H5File>(
            path.native_string().c_str(), access_mode, creation_properties,
            access_properties);
        result->flush(H5F_SCOPE_GLOBAL);
    }
    catch(H5::FileIException const& /* exception */) {
        throw IOError(path.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_READ]);
    }

    return result;
}


std::shared_ptr<H5::H5File> open_file(
    Path const& path,
    OpenMode open_mode)
{
    std::shared_ptr<H5::H5File> result;

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
