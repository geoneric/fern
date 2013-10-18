#include "geoneric/io/core/dataset.h"


namespace geoneric {

Dataset::Dataset(
    String const& name,
    OpenMode open_mode)

    : _name(name),
      _open_mode(open_mode)

{
}


String const& Dataset::name() const
{
    return _name;
}


OpenMode Dataset::open_mode() const
{
    return _open_mode;
}

} // namespace geoneric
