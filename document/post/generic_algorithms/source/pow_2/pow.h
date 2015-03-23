#pragma once
#include <cmath>


template<
    typename Base,
    typename Exponent,
    typename Result>
void pow(
    Base const& base,
    Exponent const& exponent,
    Result& result)
{
    size_t const nr_rows{base.shape()[0]};
    size_t const nr_cols{base.shape()[1]};

    for(size_t r = 0; r < nr_rows; ++r) {
        for(size_t c = 0; c < nr_cols; ++c) {
            result[r][c] = std::pow(base[r][c], exponent);
        }
    }
}
