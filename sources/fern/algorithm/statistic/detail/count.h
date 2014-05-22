#pragma once


namespace fern {
namespace count {
namespace detail {
namespace dispatch {

template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ArrayCollectionCategory>
class Count
{
};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Count<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_0d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Count()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Count(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 0d array
    inline void calculate(
        Values const& values,
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        if(!InputNoDataPolicy::is_no_data()) {
            fern::get(result) = fern::get(values) == fern::get(value) ? 1 : 0;
        }
        else {
            OutputNoDataPolicy::mark_as_no_data();
        }
    }

};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Count<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_1d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Count()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Count(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 1d array
    inline void calculate(
        Values const& values,
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        size_t const size = fern::size(values);

        auto ranges = IndexRanges<1>{
            IndexRange(0, size)
        };

        calculate(ranges, values, value, result);
    }

    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Values const& values,
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        size_t const begin = indices[0].begin();
        size_t const end = indices[0].end();
        bool data_seen{false};

        if(begin < end) {

            typename ArgumentTraits<Result>::value_type& count =
                fern::get(result);
            count = 0;

            for(size_t i = begin; i < end; ++i) {

                if(!InputNoDataPolicy::is_no_data(i)) {

                    count += fern::get(values, i) == value ? 1 : 0;
                    data_seen = true;
                }
            }
        }

        if(!data_seen) {
            OutputNoDataPolicy::mark_as_no_data();
        }
    }

};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Count<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_2d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Count()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Count(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 2d array
    inline void calculate(
        Values const& values,
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        size_t const size1 = fern::size(values, 0);
        size_t const size2 = fern::size(values, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, values, value, result);
    }

    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Values const& values,
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        size_t const begin1 = indices[0].begin();
        size_t const end1 = indices[0].end();
        size_t const begin2 = indices[1].begin();
        size_t const end2 = indices[1].end();
        bool data_seen{false};

        if(begin1 < end1 && begin2 < end2) {

            typename ArgumentTraits<Result>::value_type& count =
                fern::get(result);
            count = 0;

            for(size_t i = begin1; i < end1; ++i) {
                for(size_t j = begin2; j < end2; ++j) {

                    if(!InputNoDataPolicy::is_no_data(i, j)) {

                        count += fern::get(values, i, j) == value ? 1 : 0;
                        data_seen = true;
                    }
                }
            }
        }

        if(!data_seen) {
            OutputNoDataPolicy::mark_as_no_data();
        }
    }

};

} // namespace dispatch
} // namespace detail
} // namespace count
} // namespace fern
