#include "geoneric/io/core/dataset.h"


namespace geoneric {

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

} // namespace geoneric
