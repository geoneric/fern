#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <vector>
#include "fern/algorithm/core/index_range.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      The IndexRanges class represents a collection of index ranges
                for multiple dimensions.
    @tparam     nr_dimensions Number of dimensions to store an index range for.
    @sa         IndexRange

    This is useful for defining a multi-dimensional subset of (array)
    dimensions.
*/
template<
    size_t nr_dimensions>
class IndexRanges:
    private std::array<IndexRange, nr_dimensions>

{

public:

    //! Default constuct an instance.
                   IndexRanges         ()=default;

                   IndexRanges         (IndexRanges const& other);

                   IndexRanges         (IndexRanges&& other);

                   IndexRanges         (IndexRange range);

                   IndexRanges         (IndexRange range1,
                                        IndexRange range2);

    IndexRanges&   operator=           (IndexRanges const& other)=default;

    IndexRanges&   operator=           (IndexRanges&& other);

                   ~IndexRanges        ()=default;

    IndexRange const&
                   operator[]          (size_t index) const;

    IndexRange&    operator[]          (size_t index);

    bool           empty               () const;

    size_t         size                () const;

private:

    using Base = std::array<IndexRange, nr_dimensions>;

};


//! Copy construct an inѕtance.
/*!
*/
template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>::IndexRanges(
    IndexRanges const& other)

    : Base(other)

{
}


//! Move construct and instance.
/*!
*/
template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>::IndexRanges(
    IndexRanges&& other)

    : Base(std::move(other))

{
}


//! Construct an instance based on a single index range.
/*!
  \param     range Index range to use for the dimension.
  \warning   \a nr_dimensions must be 1.
*/
template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>::IndexRanges(
    IndexRange range)

    :  Base{{range}}

{
    static_assert(nr_dimensions == 1, "");
}


//! Construct an instance based on two index ranges.
/*!
  \param     range1 Index range to use for first dimension.
  \param     range2 Index range to use for second dimension.
  \warning   \a nr_dimensions must be 2.
*/
template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>::IndexRanges(
    IndexRange range1,
    IndexRange range2)

    :  Base{{range1, range2}}

{
    static_assert(nr_dimensions == 2, "");
}


//! Move assign \a other to this instance.
/*!
*/
template<
    size_t nr_dimensions>
inline IndexRanges<nr_dimensions>& IndexRanges<nr_dimensions>::operator=(
    IndexRanges<nr_dimensions>&& other)
{
    Base::operator=(std::move(other));
    return *this;
}


//! Subscript instance by \a index.
/*!
  \return    The index range of dimension \a index.
  \exception \a index must be smaller than \a nr_dimensions.
*/
template<
    size_t nr_dimensions>
inline IndexRange const& IndexRanges<nr_dimensions>::operator[](
    size_t index) const
{
    assert(index < nr_dimensions);
    return Base::operator[](index);
}


//! Subscript instance by \a index.
/*!
  \return    The index range of dimension \a index.
  \exception \a index must be smaller than \a nr_dimensions.
*/
template<
    size_t nr_dimensions>
inline IndexRange& IndexRanges<nr_dimensions>::operator[](
    size_t index)
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
inline size_t IndexRanges<nr_dimensions>::size() const
{
    size_t result = 0;

    for(auto const& range: *this) {
        if(range.empty()) {
            result = 0;
            break;
        }
        else {
            result += range.size();
        }
    }

    return result;
}


//! Return whether \a lhs and \a rhs are equal.
/*!
*/
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


//! Return whether \a lhs and \a rhs are not equal.
/*!
*/
template<
    size_t nr_dimensions>
inline bool operator!=(
    IndexRanges<nr_dimensions> const& lhs,
    IndexRanges<nr_dimensions> const& rhs)
{
    return !(lhs == rhs);
}


//! Write \a ranges to output \a stream.
/*!
*/
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


std::vector<IndexRanges<1>>
                   index_ranges        (size_t const nr_blocks,
                                        size_t const size1);

std::vector<IndexRanges<2>>
                   index_ranges        (size_t const nr_blocks,
                                        size_t const size1,
                                        size_t const size2);

} // namespace algorithm
} // namespace fern
