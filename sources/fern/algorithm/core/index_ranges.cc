#include "fern/algorithm/core/index_ranges.h"


namespace fern {

//! Determine index ranges to use for partitioning a 2D array using \a nr_threads available worker threads.
/*!
  \param     nr_threads Number of worker threads.
  \param     size1 Amount of values in the first dimension.
  \param     size2 Amount of values in the second dimension.
  \return    Collection of IndexRanges instances of partitioned blocks.

  The size of the collection returned determins on how well the 2D array can
  be devided. In case the array is empty, the collection is empty. In case the
  array is small, the size of the collection is 1. Otherwise the size is equal
  to \a nr_threads passed in or at most one more.
*/
std::vector<IndexRanges<2>> index_ranges(
    size_t const nr_threads,
    size_t const size1,
    size_t const size2)
{
    // Assuming the last size is the size of the dimension whose elements
    // are adjacent to each other in memory.
    size_t const block_width = size2;
    size_t const block_height = size1 / nr_threads;  // Integer division.
    size_t const nr_blocks = block_height > 0 ? size1 / block_height : 0;
    std::vector<IndexRanges<2>> ranges(nr_blocks);

    size_t row_offset = 0;
    for(size_t i = 0; i < nr_blocks; ++i) {
        ranges[i] = IndexRanges<2>(
            IndexRange(row_offset, row_offset + block_height),
            IndexRange(0, block_width));
        row_offset += block_height;
    }

    // Handle the remaining values.
    size_t const block_height_remainder = size1 - nr_blocks * block_height;

    if(block_height_remainder > 0) {
        ranges.emplace_back(
            IndexRange(row_offset, row_offset + block_height_remainder),
            IndexRange(0, block_width));
    }

    return std::move(ranges);
}

} // namespace fern
