// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/interpreter/data_source.h"


namespace fern {

DataSource::DataSource()

#ifndef NDEBUG
    : _read(false)
#endif

{
}


#ifndef NDEBUG
bool DataSource::data_has_been_read_already() const
{
    return _read;
}


void DataSource::set_data_has_been_read() const
{
    assert(!data_has_been_read_already());
    _read = true;
}
#endif

} // namespace fern
