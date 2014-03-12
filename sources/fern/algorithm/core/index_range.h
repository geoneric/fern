#pragma once
#include <cstddef>
#include <iostream>


namespace fern {

//! The IndexRange class represents a range of (array) indices for one (array) dimension.
/*!
  This is useful for defining a subset of a (array) dimension.

  A range is defined by a begin index and an end index. The end index is
  outside of the range of valid indices. An empty range is a range whose
  begin and end indices are equal. Unless a range is empty, a begin index is
  always smaller than an end index.
*/
class IndexRange
{

public:

    typedef size_t index;

                   IndexRange          ();

                   IndexRange          (index begin,
                                        index end);

    index          begin               () const;

    index          end                 () const;

    bool           empty               () const;

private:

    index          _begin;

    index          _end;

};


bool               operator==          (IndexRange const& lhs,
                                        IndexRange const& rhs);

bool               operator!=          (IndexRange const& lhs,
                                        IndexRange const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        IndexRange const& range);

} // namespace fern
