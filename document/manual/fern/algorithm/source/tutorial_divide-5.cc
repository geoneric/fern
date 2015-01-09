#include <algorithm>
#include <cassert>
#include <cstdlib>
#include "fern/core/vector_traits.h"
#include "fern/algorithm/algebra/elementary/divide.h"
#include "fern/algorithm/policy/detect_no_data_by_value.h"


int main()
{
    namespace fa = fern::algorithm;

    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<std::vector<double>>,
        fa::DetectNoDataByValue<std::vector<double>>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<std::vector<double>>;

    double const no_data{-999.0};

    std::vector<double> value1 = { 1.0, 2.0, 3.0, 4.0, no_data };
    std::vector<double> value2 = { 0.0, 4.0, no_data, 2.0, 1.0 };
    std::vector<double> result(value1.size());

    InputNoDataPolicy input_no_data_policy{{value1, no_data},
        {value2, no_data}};
    OutputNoDataPolicy output_no_data_policy(result, no_data);

    fa::algebra::divide<fa::divide::OutOfDomainPolicy,
        fa::divide::OutOfRangePolicy>(input_no_data_policy,
            output_no_data_policy, fa::sequential, value1, value2, result);

    assert(result[0] == no_data);  // Zero in second argument.
    assert(result[1] == 2.0 / 4.0);
    assert(result[2] == no_data);  // No-data in second argument.
    assert(result[3] == 4.0 / 2.0);
    assert(result[4] == no_data);  // No-data in first argument.

    return EXIT_SUCCESS;
}
