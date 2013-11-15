#include "fern/io/core/dataset.h"


namespace fern {

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

} // namespace fern
