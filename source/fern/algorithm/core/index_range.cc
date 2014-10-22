#include "fern/algorithm/core/index_range.h"
#include <cassert>


namespace fern {
namespace algorithm {

//! Default construct an IndexRange instance.
/*!
    The begin and end indices are default initialized.
*/
IndexRange::IndexRange()

    : _begin(),
      _end()

{
}


//! Construct an IndexRange instance.
/*!
  \param     begin Begin index of range.
  \param     end End index of range.
  \warning   \a begin must be >= \a end
*/
IndexRange::IndexRange(
    index begin,
    index end)

    : _begin(begin),
      _end(end)

{
    assert(_begin <= _end);
}


//! Return whether or not the range is empty.
/*!
    \sa        size()

    The range is considered empty if the end index equals the begin index.
*/
bool IndexRange::empty() const
{
    return _begin == _end;
}


//! Return the number of elements in the range.
/*!
    \sa        empty()
*/
size_t IndexRange::size() const
{
    return _end - _begin;
}


//! Return whether or not \a lhs equals \a rhs.
/*!
*/
bool operator==(
    IndexRange const& lhs,
    IndexRange const& rhs)
{
    return
        lhs.begin() == rhs.begin() &&
        lhs.end() == rhs.end()
        ;
}


//! Return whether or not \a lhs does not equal \a rhs.
/*!
*/
bool operator!=(
    IndexRange const& lhs,
    IndexRange const& rhs)
{
    return
        lhs.begin() != rhs.begin() ||
        lhs.end() != rhs.end()
        ;
}


//! Write \a range to an output \a stream.
/*!
*/
std::ostream& operator<<(
    std::ostream& stream,
    IndexRange const& range)
{
    stream << "[" << range.begin() << ", " << range.end() << ")";
    return stream;
}

} // namespace algorithm
} // namespace fern
