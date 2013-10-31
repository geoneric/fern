#include "geoneric/io/core/file.h"
#include <boost/filesystem.hpp>


namespace geoneric {

//! Return whether \a path points to a regular file or a symbolic link pointing to a regular file.
/*!
  \param     path Path of file to check.
*/
bool file_exists(
    Path const& path)
{
    namespace fs = boost::filesystem;

    fs::path pathname(path.native_string().encode_in_default_encoding());
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


//! Return whether \a path points to a writable directory or a symbolic link pointing to a writable directory.
/*!
  \tparam    path Path of directory to check.
*/
bool directory_is_writable(
    Path const& path)
{
    namespace fs = boost::filesystem;

    fs::path pathname(path.is_empty() ? "."
        : path.native_string().encode_in_default_encoding());
    fs::file_status status(fs::status(pathname));

    if(fs::exists(status)) {
        if(fs::is_directory(status)) {
            return true;
        }

        pathname = fs::canonical(pathname);
        status = fs::status(pathname);

        return fs::exists(status) && fs::is_directory(status);
    }

    return false;
}

} // namespace geoneric
