#pragma once


namespace fern {
namespace convolve {

class ReplaceNoDataByFocalAverage
{

public:

    template<class InputNoDataPolicy, class SourceImage, class Value>
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
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
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
    for(size_t r = first_row; r <= last_row; ++r) {
        for(size_t c = first_col; c <= last_col; ++c) {
            if(!(r == row && c == col)) {
                if(!input_no_data_policy.is_no_data(r, c)) {
                    value += get(source, r, c);
                    ++count;
                }
            }
        }
    }

    // At least the focal cell must be non-no-data. Otherwise it would have
    // been skipped entirely and we wouldn't be here.
    assert(count > 0u);
    value /= count;

    return true;
}

} // namespace convolve
} // namespace fern
