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
