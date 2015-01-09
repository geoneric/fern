#pragma once
#include <cassert>
#include <cstddef>


namespace fern {
namespace algorithm {
namespace convolve {

/*!
    @brief      AlternativeForNoDataPolicy which replaces no-data values
                by the focal average.
    @sa         @ref fern_algorithm_convolution_policies
*/
class ReplaceNoDataByFocalAverage
{

public:

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
    static bool    value               (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const nr_rows,
                                        size_t const nr_cols,
                                        size_t const row,
                                        size_t const col,
                                        Value& value);

};


template<
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value>
inline bool ReplaceNoDataByFocalAverage::value(
        InputNoDataPolicy const& input_no_data_policy,
        SourceImage const& source,
        size_t const nr_rows,
        size_t const nr_cols,
        size_t const row,
        size_t const col,
        Value& value)
{
    // Use average of cells surrounding current cell as the value.

    value = Value{0};
    size_t const first_row = row > 0u ? row - 1 : row;
    size_t const first_col = col > 0u ? col - 1 : col;
    size_t const last_row = row < nr_rows - 1 ? row + 1 : row;
    size_t const last_col = col < nr_cols - 1 ? col + 1 : col;

    size_t count = 0;

    size_t index_;

    for(size_t r = first_row; r <= last_row; ++r) {

        index_ = index(source, r, first_col);

        for(size_t c = first_col; c <= last_col; ++c) {

            if(!(r == row && c == col)) {
                if(!std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    value += get(source, index_);
                    ++count;
                }
            }

            ++index_;
        }
    }

    // At least the focal cell must be non-no-data. Otherwise it would have
    // been skipped entirely and we wouldn't be here.
    // -> But in case the radius of the convolution kernel is larger than 1,
    // the focal cell is not surrounding cell row, col. It is possible that
    // cell row, col is surrounded by only no-data.
    // TODO Handle this once this assertion fails for the first time.
    assert(count > 0u);
    value /= count;

    return true;
}

} // namespace convolve
} // namespace algorithm
} // namespace fern
