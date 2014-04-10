#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include "fern/algorithm/core/index_range.h"


namespace fern {

template<
    size_t nr_dimensions>
class IndexRanges:
    private std::array<IndexRange, nr_dimensions>

{

public:

                   IndexRanges         ()=default;

                   IndexRanges         (IndexRanges const& other);

                   IndexRanges         (IndexRanges&& other);

                   IndexRanges         (IndexRange range);

                   IndexRanges         (IndexRange range1,
                                        IndexRange range2);

    IndexRanges&   operator=           (IndexRanges const& other)=delete;

    IndexRanges&   operator=           (IndexRanges&& other);

                   ~IndexRanges        ()=default;

    IndexRange const&
                   operator[]          (size_t index) const;

    bool           empty               () const;

private:

    typedef std::array<IndexRange, nr_dimensions> Base;

};


template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>::IndexRanges(
    IndexRanges const& other)

    : Base(other)

{
}


template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>::IndexRanges(
    IndexRanges&& other)

    : Base(std::move(other))

{
}


template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>::IndexRanges(
    IndexRange range)

    :  Base{{range}}

{
    static_assert(nr_dimensions == 1, "");
}


template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>::IndexRanges(
    IndexRange range1,
    IndexRange range2)

    :  Base{{range1, range2}}

{
    static_assert(nr_dimensions == 2, "");
}


template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>& IndexRanges<nr_dimensions>::operator=(
    IndexRanges<nr_dimensions>&& other)
{
    Base::operator=(std::move(other));
    return *this;
}


template<
    size_t nr_dimensions>
inline IndexRange const& IndexRanges<nr_dimensions>::operator[](
    size_t index) const
{
    assert(index < nr_dimensions);
    return Base::operator[](index);
}


//! Return whether the index ranges are empty.
/*!
  The ranges are considered empty if at least one of the layered index ranges
  is empty.
*/
template<
    size_t nr_dimensions>
inline bool IndexRanges<nr_dimensions>::empty() const
{
    return std::any_of(this->cbegin(), this->cend(),
        [](IndexRange const& range){ return range.empty(); });
}


template<
    size_t nr_dimensions>
inline bool operator==(
    IndexRanges<nr_dimensions> const& lhs,
    IndexRanges<nr_dimensions> const& rhs)
{
    for(size_t i = 0; i < nr_dimensions; ++i) {
        if(lhs[i] != rhs[i]) {
            return false;
        }
    }

    return true;
}


template<
    size_t nr_dimensions>
inline bool operator!=(
    IndexRanges<nr_dimensions> const& lhs,
    IndexRanges<nr_dimensions> const& rhs)
{
    return !(lhs == rhs);
}


template<
    size_t nr_dimensions>
inline std::ostream& operator<<(
    std::ostream& stream,
    IndexRanges<nr_dimensions> const& ranges)
{
    stream << "[";

    for(size_t i = 0; i < nr_dimensions; ++i) {
        stream << ranges[i];
    }

    stream << "]";

    return stream;
}


std::vector<IndexRanges<2>>
                   index_ranges        (size_t const nr_blocks,
                                        size_t const size1,
                                        size_t const size2);

} // namespace fern
