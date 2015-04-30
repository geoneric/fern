// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/core/file.h"
#include <fstream>
#include <boost/filesystem.hpp>


namespace fern {
namespace language {

//! Return whether \a path points to a regular file or a symbolic link pointing to a regular file.
/*!
  \param     path Path of file to check.
*/
bool file_exists(
    Path const& path)
{
    namespace fs = boost::filesystem;

    fs::path pathname(path.native_string());
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

    fs::path pathname(path.native_string());
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

    fs::path pathname(path.is_empty() ? "." : path.native_string());
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
    std::string const& value,
    Path const& path)
{
    std::ofstream stream(path.native_string());
    stream << value;
    stream.flush();

    if(!stream.good()) {
        // TODO Exception.
        assert(false);
    }
}

} // namespace language
} // namespace fern
