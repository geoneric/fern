#include "geoneric/io/geoneric/utils.h"
#include "geoneric/core/data_name.h"
#include "geoneric/core/io_error.h"


namespace geoneric {

std::shared_ptr<H5::H5File> open_file(
    Path const& pathname,
    OpenMode open_mode)
{
    assert(open_mode == OpenMode::OVERWRITE);
    std::shared_ptr<H5::H5File> file;

    try {
        unsigned int access_mode = H5F_ACC_TRUNC; // | H5F_ACC_RDWR?
        H5::FileCreatPropList creation_properties =
            H5::FileCreatPropList::DEFAULT;
        H5::FileAccPropList access_properties = H5::FileAccPropList::DEFAULT;
        file.reset(new H5::H5File(
            pathname.native_string().encode_in_default_encoding().c_str(),
            access_mode, creation_properties, access_properties));
        file->flush(H5F_SCOPE_GLOBAL);
    }
    catch(H5::FileIException const& /* exception */) {
        throw IOError(pathname.native_string(),
            Exception::messages()[MessageId::CANNOT_BE_CREATED]);
    }

    return file;
}

} // namespace geoneric
