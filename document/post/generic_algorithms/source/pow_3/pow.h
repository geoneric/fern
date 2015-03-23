#pragma once
#include <cmath>
#include "customization_point.h"


template<
    typename Base,
    typename Exponent,
    typename Result>
void pow(
    Base const& base,
    Exponent const& exponent,
    Result& result)
{
    size_t const nr_rows{size(base, 0)};
    size_t const nr_cols{size(base, 1)};

    for(size_t r = 0; r < nr_rows; ++r) {
        for(size_t c = 0; c < nr_cols; ++c) {
            get(result, r, c) = std::pow(get(base, r, c), exponent);
        }
    }
}
