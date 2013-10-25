#include "geoneric/io/geoneric/utils.h"
#include "geoneric/core/data_name.h"
#include "geoneric/core/io_error.h"


namespace geoneric {

static std::shared_ptr<H5::H5File> open_file_for_overwrite(
    Path const& pathname)
{
    std::shared_ptr<H5::H5File> result;

    try {
        unsigned int access_mode = H5F_ACC_TRUNC;
        H5::FileCreatPropList creation_properties =
            H5::FileCreatPropList::DEFAULT;
        H5::FileAccPropList access_properties = H5::FileAccPropList::DEFAULT;
        result.reset(new H5::H5File(
            pathname.native_string().encode_in_default_encoding().c_str(),
            access_mode, creation_properties, access_properties));
        result->flush(H5F_SCOPE_GLOBAL);
    }
    catch(H5::FileIException const& /* exception */) {
        throw IOError(pathname.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_CREATED]);
    }

    return result;
}


static std::shared_ptr<H5::H5File> open_file_for_update(
    Path const& pathname)
{
    std::shared_ptr<H5::H5File> result;

    try {
        unsigned int access_mode = H5F_ACC_RDWR;
        H5::FileCreatPropList creation_properties =
            H5::FileCreatPropList::DEFAULT;
        H5::FileAccPropList access_properties = H5::FileAccPropList::DEFAULT;
        result.reset(new H5::H5File(
            pathname.native_string().encode_in_default_encoding().c_str(),
            access_mode, creation_properties, access_properties));
        result->flush(H5F_SCOPE_GLOBAL);
    }
    catch(H5::FileIException const& /* exception */) {
        throw IOError(pathname.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_WRITTEN]);
    }

    return result;
}


static std::shared_ptr<H5::H5File> open_file_for_read(
    Path const& pathname)
{
    std::shared_ptr<H5::H5File> result;

    try {
        unsigned int access_mode = H5F_ACC_RDONLY;
        H5::FileCreatPropList creation_properties =
            H5::FileCreatPropList::DEFAULT;
        H5::FileAccPropList access_properties = H5::FileAccPropList::DEFAULT;
        result.reset(new H5::H5File(
            pathname.native_string().encode_in_default_encoding().c_str(),
            access_mode, creation_properties, access_properties));
        result->flush(H5F_SCOPE_GLOBAL);
    }
    catch(H5::FileIException const& /* exception */) {
        throw IOError(pathname.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_READ]);
    }

    return result;
}


std::shared_ptr<H5::H5File> open_file(
    Path const& pathname,
    OpenMode open_mode)
{
    std::shared_ptr<H5::H5File> result;

    switch(open_mode) {
        case OpenMode::OVERWRITE: {
            result = open_file_for_overwrite(pathname);
            break;
        }
        case OpenMode::UPDATE: {
            result = open_file_for_update(pathname);
            break;
        }
        case OpenMode::READ: {
            result = open_file_for_read(pathname);
            break;
        }
    }

    return result;
}

} // namespace geoneric
