#include "Ranally/IO/Dataset.h"


namespace ranally {

Dataset::Dataset(
    String const& name)

    : _name(name)

{
}


Dataset::~Dataset()
{
}


String const& Dataset::name() const
{
    return _name;
}

} // namespace ranally
