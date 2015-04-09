// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/compiler/data_description.h"


namespace fern {

DataDescription::DataDescription(
    String const& name)

    : _name(name)

{
    // TODO Pimp argument description with types.
    _description = _name;
}


std::string DataDescription::name() const
{
    return _name.encode_in_default_encoding();
}


std::string DataDescription::description() const
{
    return _description.encode_in_default_encoding();
}

} // namespace fern
