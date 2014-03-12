#include "fern/algorithm/core/index_range.h"
#include <cassert>


namespace fern {

IndexRange::IndexRange()

    : _begin(),
      _end()

{
}


IndexRange::IndexRange(
    index begin,
    index end)

    : _begin(begin),
      _end(end)

{
    assert(_begin <= _end);
}



//! Return the first index in the range.
/*!
*/
IndexRange::index IndexRange::begin() const
{
    return _begin;
}


//! Return the index past the last index in the range.
/*!
*/
IndexRange::index IndexRange::end() const
{
    return _end;
}


bool IndexRange::empty() const
{
    return _begin == _end;
}


bool operator==(
    IndexRange const& lhs,
    IndexRange const& rhs)
{
    return
        lhs.begin() == rhs.begin() &&
        lhs.end() == rhs.end()
        ;
}


bool operator!=(
    IndexRange const& lhs,
    IndexRange const& rhs)
{
    return
        lhs.begin() != rhs.begin() ||
        lhs.end() != rhs.end()
        ;
}


std::ostream& operator<<(
    std::ostream& stream,
    IndexRange const& range)
{
    stream << "[" << range.begin() << ", " << range.end() << ")";
    return stream;
}

} // namespace fern
