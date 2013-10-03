#include "geoneric/io/core/file.h"
#include <boost/filesystem.hpp>


namespace geoneric {

//! Return whether \a filename is the name of a regular file or a symbolic link pointing to a regular file.
/*!
  \param     filename Name of file to check.
  \return    true or false
*/
bool file_exists(
    String const& filename)
{
    namespace fs = boost::filesystem;

    fs::path pathname(filename.encode_in_default_encoding());
    fs::file_status status(fs::status(pathname));

    if(fs::exists(status)) {
        if(fs::is_regular_file(status)) {
            return true;
        }

        pathname = fs::canonical(pathname);
        status = fs::status(pathname);

        return fs::exists(status) && fs::is_regular_file(status);
    }

    return false;
}

} // namespace geoneric
