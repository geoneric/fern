// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>
#include <iostream>


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      The IndexRange class represents a range of (array)
                indices for one (array) dimension.

    This is useful for defining a subset of a (array) dimension.

    A range is defined by a begin index and an end index. The end index
    is outside of the range of valid indices. An empty range is a range
    whose begin and end indices are equal. Unless a range is empty,
    a begin index is always smaller than an end index.
*/
class IndexRange
{

public:

    using index = size_t;

                   IndexRange          ();

                   IndexRange          (index begin,
                                        index end);

    index          begin               () const;

    index          end                 () const;

    bool           empty               () const;

    size_t         size                () const;

    void           set_end             (index end);

private:

    index          _begin;

    index          _end;

};


// begin() and end() are used in for loops over arrays. They must be inlined.


//! Return the first index in the range.
/*!
*/
inline IndexRange::index IndexRange::begin() const
{
    return _begin;
}


//! Return the index past the last index in the range.
/*!
*/
inline IndexRange::index IndexRange::end() const
{
    return _end;
}


bool               operator==          (IndexRange const& lhs,
                                        IndexRange const& rhs);

bool               operator!=          (IndexRange const& lhs,
                                        IndexRange const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        IndexRange const& range);

} // namespace algorithm
} // namespace fern
