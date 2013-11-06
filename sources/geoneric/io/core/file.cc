#include "geoneric/io/core/file.h"
#include <fstream>
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


//! Return whether \a path points to a writable file or a symbolic link pointing to a writable file.
/*!
  \param     path Path of file to check.
*/
bool file_is_writable(
    Path const& path)
{
    namespace fs = boost::filesystem;

    fs::path pathname(path.native_string().encode_in_default_encoding());
    fs::file_status status(fs::status(pathname));

    if(!fs::exists(status)) {
        return false;
    }

    if(fs::is_regular_file(status) && status.permissions() & fs::owner_write) {
        return true;
    }

    pathname = fs::canonical(pathname);
    status = fs::status(pathname);

    return fs::exists(status) && fs::is_regular_file(status) &&
        status.permissions() & fs::owner_write;
}


//! Return whether \a path points to a writable directory or a symbolic link pointing to a writable directory.
/*!
  \param     path Path of directory to check.
*/
bool directory_is_writable(
    Path const& path)
{
    namespace fs = boost::filesystem;

    fs::path pathname(path.is_empty() ? "."
        : path.native_string().encode_in_default_encoding());
    fs::file_status status(fs::status(pathname));

    if(!fs::exists(status)) {
        return false;
    }

    if(fs::is_directory(status) && status.permissions() & fs::owner_write) {
        return true;
    }

    pathname = fs::canonical(pathname);
    status = fs::status(pathname);

    return fs::exists(status) && fs::is_directory(status) &&
        status.permissions() & fs::owner_write;
}


void write_file(
    String const& value,
    Path const& path)
{
    std::ofstream stream(path.native_string().encode_in_default_encoding());
    stream << value.encode_in_utf8();
    stream.flush();

    if(!stream.good()) {
        // TODO Exception.
        assert(false);
    }
}

} // namespace geoneric
