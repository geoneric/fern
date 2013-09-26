#include "geoneric/io/core/path.h"


namespace geoneric {

Path::Path(
    boost::filesystem::path const& path)

    : boost::filesystem::path(path)

{
}


Path::Path(
    String const& string)

    : boost::filesystem::path(string.encode_in_default_encoding())

{
}


Path Path::stem() const
{
    return Path(boost::filesystem::path::stem());
}


bool Path::operator==(
    Path const& path) const
{
    return dynamic_cast<boost::filesystem::path const&>(*this) ==
        dynamic_cast<boost::filesystem::path const&>(path);
}


Path::operator String() const
{
    return String(native());
}

} // namespace geoneric
