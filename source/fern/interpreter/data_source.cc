#include "fern/interpreter/data_source.h"


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
