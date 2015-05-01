// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/core/dataset.h"


namespace fern {
namespace language {

Dataset::Dataset(
    std::string const& name,
    OpenMode open_mode)

    : _name(name),
      _open_mode(open_mode)

{
}


std::string const& Dataset::name() const
{
    return _name;
}


OpenMode Dataset::open_mode() const
{
    return _open_mode;
}

} // namespace language
} // namespace fern
