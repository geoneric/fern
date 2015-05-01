// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/path.h"
#include "fern/core/string.h"


namespace fern {

Path::Path(
    char const* string)

    : boost::filesystem::path(string)

{
}


Path::Path(
    std::string const& string)

    : boost::filesystem::path(string)

{
}


Path Path::stem() const
{
    return Path(boost::filesystem::path::stem().generic_string());
}


bool Path::operator==(
    Path const& path) const
{
    return dynamic_cast<boost::filesystem::path const&>(*this) ==
        dynamic_cast<boost::filesystem::path const&>(path);
}


//! Returns the path as a string, formatted according to the generic conventions.
/*!
  \sa        native_string()
*/
std::string Path::generic_string() const
{
    return boost::filesystem::path::generic_string();
}


//! Returns the path as a string, formatted according to the native conventions.
/*!
  \sa        generic_string()
*/
std::string Path::native_string() const
{
    return native();
}


bool Path::is_empty() const
{
    return boost::filesystem::path::empty();
}


bool Path::is_absolute() const
{
    return boost::filesystem::path::is_absolute();
}


std::vector<std::string> Path::names() const
{
    return split(generic_string(), "/");
}


Path Path::parent_path() const
{
    return Path(boost::filesystem::path::parent_path().generic_string());
}


Path Path::filename() const
{
    return Path(boost::filesystem::path::filename().generic_string());
}


Path& Path::replace_extension(
    Path const& extension)
{
    boost::filesystem::path::replace_extension(extension);
    return *this;
}


Path& Path::operator/=(
    Path const& path)
{
    boost::filesystem::path::operator/=(path);
    return *this;
}


Path operator/(
    Path const& lhs,
    Path const& rhs)
{
    Path result(lhs);
    return result /= rhs;
}


std::ostream& operator<<(
    std::ostream& stream,
    Path const& path)
{
    stream << path.generic_string();
    return stream;
}

} // namespace fern
