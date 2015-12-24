// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/array_2d_traits.h"
#include "fern/core/base_class.h"
#include "fern/core/data_customization_point.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace algorithm {
namespace if_ {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_then_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    if(!std::get<0>(input_no_data_policy).is_no_data() && get(condition) &&
            !std::get<1>(input_no_data_policy).is_no_data()) {
        get(result) = get(true_value);
    }
    else {
        output_no_data_policy.mark_as_no_data();
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_then_0d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    size_t index_;

    if(!std::get<0>(input_no_data_policy).is_no_data() && get(condition)) {

        // Condition is not masked and condition is true.
        // Copy all cells from true_value to result.

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(!std::get<1>(input_no_data_policy).is_no_data(index_)) {
                    get(result, index_) = get(true_value, index_);
                }
                else {
                    output_no_data_policy.mark_as_no_data(index_);
                }

                ++index_;
            }
        }
    }
    else {

        // Condition is masked or condition is false.
        // Mask all cells in result.
        // TODO Use something like this, to allow the policy to use a faster
        //      method to mask the whole region.
        //      output_no_data_policy.mark_as_no_data(index_ranges);

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                output_no_data_policy.mark_as_no_data(index_);

                ++index_;
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_then_2d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    size_t index_;

    if(std::get<1>(input_no_data_policy).is_no_data()) {
        // true_value is no-data.
        // It doesn't matter whether or not the condition contains no-data
        // or is false. The result has to be filled with no-data.
        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                output_no_data_policy.mark_as_no_data(index_);
                ++index_;
            }
        }
    }
    else {
        // true_value is not no-data.
        const_reference<TrueValue> tv(get(true_value));

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(!std::get<0>(input_no_data_policy).is_no_data(index_) &&
                        get(condition, index_)) {
                    get(result, index_) = tv;
                }
                else {
                    output_no_data_policy.mark_as_no_data(index_);
                }

                ++index_;
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_then_2d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    size_t index_;

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_ = index(result, i, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!std::get<0>(input_no_data_policy).is_no_data(index_) &&
                    get(condition, index_) &&
                    !std::get<1>(input_no_data_policy).is_no_data(index_)) {
                get(result, index_) = get(true_value, index_);
            }
            else {
                output_no_data_policy.mark_as_no_data(index_);
            }

            ++index_;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_0d_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    if(std::get<0>(input_no_data_policy).is_no_data()) {
        output_no_data_policy.mark_as_no_data();
    }
    else if(get(condition)) {
        if(std::get<1>(input_no_data_policy).is_no_data()) {
            output_no_data_policy.mark_as_no_data();
        }
        else {
            get(result) = get(true_value);
        }
    }
    else {
        if(std::get<2>(input_no_data_policy).is_no_data()) {
            output_no_data_policy.mark_as_no_data();
        }
        else {
            get(result) = get(false_value);
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_1d_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    if(std::get<1>(input_no_data_policy).is_no_data()) {
        if(std::get<2>(input_no_data_policy).is_no_data()) {
            // True value and false value contain no-data.
            // It doesn't matter whether or not the condition contains valid
            // values or not, and whether valid values evaluate to true or
            // not. Fill the result with no-data.
            for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                    ++i) {
                output_no_data_policy.mark_as_no_data(i);
            }
        }
        else {
            // True value contains no-data, but false value does not.
            const_reference<FalseValue> fv(get(false_value));

            for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                    ++i) {

                if(std::get<0>(input_no_data_policy).is_no_data(i) ||
                        get(condition, i)) {
                    output_no_data_policy.mark_as_no_data(i);
                }
                else {
                    get(result, i) = fv;
                }
            }
        }
    }
    else {
        if(std::get<2>(input_no_data_policy).is_no_data()) {
            // True value contains valid value, but false contains no-data.
            const_reference<TrueValue> tv(get(true_value));

            for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                    ++i) {

                if(std::get<0>(input_no_data_policy).is_no_data(i) ||
                        !get(condition, i)) {
                    output_no_data_policy.mark_as_no_data(i);
                }
                else {
                    get(result, i) = tv;
                }
            }
        }
        else {
            // True value and false value contain valid value.
            const_reference<TrueValue> tv(get(true_value));
            const_reference<FalseValue> fv(get(false_value));

            for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                    ++i) {

                if(std::get<0>(input_no_data_policy).is_no_data(i)) {
                    output_no_data_policy.mark_as_no_data(i);
                }
                else {
                    if(get(condition, i)) {
                        get(result, i) = tv;
                    }
                    else {
                        get(result, i) = fv;
                    }
                }
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_1d_1d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    if(std::get<2>(input_no_data_policy).is_no_data()) {

        // False value contains no-data.
        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            if(std::get<0>(input_no_data_policy).is_no_data(i) ||
                    !get(condition, i) ||
                    std::get<1>(input_no_data_policy).is_no_data(i)) {
                // condition contains no-data, or
                // condition is false, or
                // true value contains no-data
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                get(result, i) = get(true_value, i);
            }
        }
    }
    else {
        // False value contains valid value.
        const_reference<FalseValue> fv(get(false_value));

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            if(std::get<0>(input_no_data_policy).is_no_data(i)) {
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                if(get(condition, i)) {
                    if(std::get<1>(input_no_data_policy).is_no_data(i)) {
                        output_no_data_policy.mark_as_no_data(i);
                    }
                    else {
                        get(result, i) = get(true_value, i);
                    }
                }
                else {
                    get(result, i) = fv;
                }
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_1d_0d_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    if(std::get<1>(input_no_data_policy).is_no_data()) {
        // True value contains no-data.

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            if(std::get<0>(input_no_data_policy).is_no_data(i) ||
                    get(condition, i) ||
                    std::get<2>(input_no_data_policy).is_no_data(i)) {
                // condition contains no-data, or
                // condition is true, or
                // false value contains no-data
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                get(result, i) = get(false_value, i);
            }
        }
    }
    else {
        // True value contains valid value.
        const_reference<TrueValue> tv(get(true_value));

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            if(std::get<0>(input_no_data_policy).is_no_data(i)) {
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                if(get(condition, i)) {
                    get(result, i) = tv;
                }
                else {
                    if(std::get<2>(input_no_data_policy).is_no_data(i)) {
                        output_no_data_policy.mark_as_no_data(i);
                    }
                    else {
                        get(result, i) = get(false_value, i);
                    }
                }
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_1d_1d_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        if(std::get<0>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i);
        }
        else {
            if(get(condition, i)) {
                if(std::get<1>(input_no_data_policy).is_no_data(i)) {
                    output_no_data_policy.mark_as_no_data(i);
                }
                else {
                    get(result, i) = get(true_value, i);
                }
            }
            else {
                if(std::get<2>(input_no_data_policy).is_no_data(i)) {
                    output_no_data_policy.mark_as_no_data(i);
                }
                else {
                    get(result, i) = get(false_value, i);
                }
            }
        }

    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_0d_2d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    size_t index_;

    if(std::get<0>(input_no_data_policy).is_no_data()) {

        // Condition is masked.
        // Mask all cells in result.
        // TODO Use something like this, to allow the policy to use a faster
        //      method to mask the whole region.
        //      output_no_data_policy.mark_as_no_data(index_ranges);

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                output_no_data_policy.mark_as_no_data(index_);

                ++index_;
            }
        }
    }
    else if(get(condition)) {

        // Condition is true.
        // Copy true_value to result.

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(std::get<1>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(result, index_) = get(true_value, index_);
                }

                ++index_;
            }
        }
    }
    else {

        // Condition is false.
        // Copy false_value to result.

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(std::get<2>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(result, index_) = get(false_value, index_);
                }

                ++index_;
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_0d_0d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    size_t index_;

    if(std::get<0>(input_no_data_policy).is_no_data()) {

        // Condition is masked.
        // Mask all cells in result.
        // TODO Use something like this, to allow the policy to use a faster
        //      method to mask the whole region.
        //      output_no_data_policy.mark_as_no_data(index_ranges);

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                output_no_data_policy.mark_as_no_data(index_);

                ++index_;
            }
        }
    }
    else if(get(condition)) {

        // Condition is true.
        // Copy true_value to result.

        if(std::get<1>(input_no_data_policy).is_no_data()) {

            // true_value is masked.
            // Mask all cells in result.
            // TODO Use something like this, to allow the policy to use a faster
            //      method to mask the whole region.
            //      output_no_data_policy.mark_as_no_data(index_ranges);

            for(size_t i = index_ranges[0].begin(); i <
                    index_ranges[0].end(); ++i) {

                index_ = index(result, i, index_ranges[1].begin());

                for(size_t j = index_ranges[1].begin();
                        j < index_ranges[1].end(); ++j) {

                    output_no_data_policy.mark_as_no_data(index_);

                    ++index_;
                }
            }
        }
        else {

            // Fill result with the true_value.
            // TODO Use fill algorithm.
            //      fill(sequential, index_ranges, true_value, result);

            const_reference<TrueValue> tv(get(true_value));

            for(size_t i = index_ranges[0].begin(); i <
                    index_ranges[0].end(); ++i) {

                index_ = index(result, i, index_ranges[1].begin());

                for(size_t j = index_ranges[1].begin();
                        j < index_ranges[1].end(); ++j) {

                    get(result, index_) = tv;

                    ++index_;
                }
            }
        }
    }
    else {

        // Condition is false.
        // Copy false_value to result.

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(std::get<2>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(result, index_) = get(false_value, index_);
                }

                ++index_;
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_0d_2d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    size_t index_;

    if(std::get<0>(input_no_data_policy).is_no_data()) {

        // Condition is masked.
        // Mask all cells in result.
        // TODO Use something like this, to allow the policy to use a faster
        //      method to mask the whole region.
        //      output_no_data_policy.mark_as_no_data(index_ranges);

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                output_no_data_policy.mark_as_no_data(index_);

                ++index_;
            }
        }
    }
    else if(get(condition)) {

        // Condition is true.
        // Copy true_value to result.

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(std::get<1>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(result, index_) = get(true_value, index_);
                }

                ++index_;
            }
        }
    }
    else {

        // Condition is false.
        // Copy false_value to result.

        if(std::get<2>(input_no_data_policy).is_no_data()) {

            // false_value is masked.
            // Mask all cells in result.
            // TODO Use something like this, to allow the policy to use a faster
            //      method to mask the whole region.
            //      output_no_data_policy.mark_as_no_data(index_ranges);

            for(size_t i = index_ranges[0].begin(); i <
                    index_ranges[0].end(); ++i) {

                index_ = index(result, i, index_ranges[1].begin());

                for(size_t j = index_ranges[1].begin();
                        j < index_ranges[1].end(); ++j) {

                    output_no_data_policy.mark_as_no_data(index_);

                    ++index_;
                }
            }
        }
        else {

            // Fill result with the true_value.
            // TODO Use fill algorithm.
            //      fill(sequential, index_ranges, true_value, result);

            const_reference<FalseValue> fv(get(false_value));

            for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                    ++i) {

                index_ = index(result, i, index_ranges[1].begin());

                for(size_t j = index_ranges[1].begin();
                        j < index_ranges[1].end(); ++j) {

                    get(result, index_) = fv;

                    ++index_;
                }
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_2d_2d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    size_t index_;

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_ = index(result, i, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                output_no_data_policy.mark_as_no_data(index_);
            }
            else {
                if(get(condition, index_)) {
                    if(std::get<1>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(true_value, index_);
                    }
                }
                else {
                    if(std::get<2>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(false_value, index_);
                    }
                }
            }

            ++index_;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_2d_2d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    size_t index_;

    if(std::get<2>(input_no_data_policy).is_no_data()) {
        // False value contains no-data.

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(std::get<0>(input_no_data_policy).is_no_data(index_) ||
                        !get(condition, index_)) {
                    // condition contains no-data, or
                    // condition is false
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    if(std::get<1>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(true_value, index_);
                    }
                }

                ++index_;
            }
        }
    }
    else {
        // False value contains a valid value.
        const_reference<FalseValue> fv(get(false_value));

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    if(get(condition, index_)) {
                        if(std::get<1>(input_no_data_policy).is_no_data(
                                index_)) {
                            output_no_data_policy.mark_as_no_data(index_);
                        }
                        else {
                            get(result, index_) = get(true_value, index_);
                        }
                    }
                    else {
                        get(result, index_) = fv;
                    }
                }

                ++index_;
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_2d_0d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    size_t index_;

    if(std::get<1>(input_no_data_policy).is_no_data()) {
        // True value contains no-data.

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
            ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(std::get<0>(input_no_data_policy).is_no_data(index_) ||
                        get(condition, index_)) {
                    // condition contains no-data, or
                    // condition is true
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    if(std::get<2>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(false_value, index_);
                    }
                }

                ++index_;
            }
        }


    }
    else {
        // True value contains a valid value.
        const_reference<TrueValue> tv(get(true_value));

        for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                ++i) {

            index_ = index(result, i, index_ranges[1].begin());

            for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                    ++j) {

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    if(get(condition, index_)) {
                        get(result, index_) = tv;
                    }
                    else {
                        if(std::get<2>(input_no_data_policy).is_no_data(
                                index_)) {
                            output_no_data_policy.mark_as_no_data(index_);
                        }
                        else {
                            get(result, index_) = get(false_value, index_);
                        }
                    }
                }

                ++index_;
            }
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_2d_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    size_t index_;

    if(std::get<1>(input_no_data_policy).is_no_data()) {
        if(std::get<2>(input_no_data_policy).is_no_data()) {
            // True value and false value contain no-data.
            // Fill the result with no-data.

            for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                    ++i) {

                index_ = index(result, i, index_ranges[1].begin());

                for(size_t j = index_ranges[1].begin();
                        j < index_ranges[1].end(); ++j) {

                    output_no_data_policy.mark_as_no_data(index_);
                    ++index_;
                }
            }
        }
        else {
            // True value contains no-data, but false value contains a
            // valid value.
            const_reference<FalseValue> fv(get(false_value));

            for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                    ++i) {

                index_ = index(result, i, index_ranges[1].begin());

                for(size_t j = index_ranges[1].begin();
                        j < index_ranges[1].end(); ++j) {

                    if(std::get<0>(input_no_data_policy).is_no_data(index_) ||
                            get(condition, index_)) {
                        // condition contains no-data, or
                        // condition is true
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = fv;
                    }

                    ++index_;
                }
            }
        }
    }
    else {
        if(std::get<2>(input_no_data_policy).is_no_data()) {
            // True value contains a valid value, but false value contains
            // no-data.
            const_reference<TrueValue> tv(get(true_value));

            for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                    ++i) {

                index_ = index(result, i, index_ranges[1].begin());

                for(size_t j = index_ranges[1].begin();
                        j < index_ranges[1].end(); ++j) {

                    if(std::get<0>(input_no_data_policy).is_no_data(index_) ||
                            !get(condition, index_)) {
                        // condition contains no-data, or
                        // condition is false
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = tv;
                    }

                    ++index_;
                }
            }
        }
        else {
            // True value and false value contain valid values.
            const_reference<TrueValue> tv(get(true_value));
            const_reference<FalseValue> fv(get(false_value));

            for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end();
                    ++i) {

                index_ = index(result, i, index_ranges[1].begin());

                for(size_t j = index_ranges[1].begin();
                        j < index_ranges[1].end(); ++j) {

                    if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        if(get(condition, index_)) {
                            get(result, index_) = tv;
                        }
                        else {
                            get(result, index_) = fv;
                        }
                    }

                    ++index_;
                }
            }
        }
    }
}


namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result,
    typename ExecutionPolicy,
    typename ConditionCollectionCategory,
    typename TrueValueCollectionCategory>
class IfThenByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result,
    typename ExecutionPolicy,
    typename ConditionCollectionCategory,
    typename TrueValueCollectionCategory,
    typename FalseValueCollectionCategory>
struct IfThenElseByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result,
    typename ExecutionPolicy>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    ExecutionPolicy,
    array_0d_tag,
    array_0d_tag>
{

    // if(0d, 0d, 0d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        if_then_0d_0d(input_no_data_policy, output_no_data_policy,
            condition, true_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result,
    typename ExecutionPolicy>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ExecutionPolicy,
    array_0d_tag,
    array_0d_tag,
    array_0d_tag>
{

    // if(0d, 0d, 0d, 0d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        if_then_else_0d_0d_0d(input_no_data_policy, output_no_data_policy,
            condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag,
    array_0d_tag,
    array_0d_tag>
{

    // if(1d, 0d, 0d, 1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value) == size(condition));

        if_then_else_1d_0d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(condition))}, condition,
            true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag,
    array_0d_tag,
    array_0d_tag>
{

    // if(1d, 0d, 0d, 1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(result) == size(condition));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(condition);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_1d_0d_0d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag,
    array_1d_tag,
    array_0d_tag>
{

    // if(1d, 1d, 0d, 1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value) == size(condition));
        assert(size(result) == size(condition));

        if_then_else_1d_1d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(condition))}, condition,
            true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag,
    array_1d_tag,
    array_0d_tag>
{

    // if(1d, 1d, 0d, 1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value) == size(condition));
        assert(size(result) == size(condition));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(condition);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_1d_1d_0d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag,
    array_0d_tag,
    array_1d_tag>
{

    // if(1d, 0d, 1d, 1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(false_value) == size(condition));
        assert(size(result) == size(condition));

        if_then_else_1d_0d_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(condition))}, condition,
            true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag,
    array_0d_tag,
    array_1d_tag>
{

    // if(1d, 0d, 1d, 1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(false_value) == size(condition));
        assert(size(result) == size(condition));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(condition);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_1d_0d_1d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag,
    array_1d_tag,
    array_1d_tag>
{

    // if(1d, 1d, 1d, 1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value) == size(condition));
        assert(size(false_value) == size(condition));
        assert(size(result) == size(condition));

        if_then_else_1d_1d_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(condition))}, condition,
            true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag,
    array_1d_tag,
    array_1d_tag>
{

    // if(1d, 1d, 1d, 1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value) == size(condition));
        assert(size(false_value) == size(condition));
        assert(size(result) == size(condition));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(condition);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_1d_1d_1d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    SequentialExecutionPolicy,
    array_0d_tag,
    array_2d_tag>
{

    // if(0d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(result, 0) == size(true_value, 0));
        assert(size(result, 1) == size(true_value, 1));

        if_then_0d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1)),
            }, condition, true_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    ParallelExecutionPolicy,
    array_0d_tag,
    array_2d_tag>
{

    // if(0d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(result, 0) == size(true_value, 0));
        assert(size(result, 1) == size(true_value, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_0d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_0d_tag>
{

    // if(2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_2d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_0d_tag>
{

    // if(2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_2d_0d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_0d_tag,
    array_0d_tag>
{

    // if(2d, 0d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_else_2d_0d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_0d_tag,
    array_0d_tag>
{

    // if(2d, 0d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_2d_0d_0d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_2d_tag>
{

    // if(2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_2d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_2d_tag>
{

    // if(2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_2d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_0d_tag,
    array_2d_tag,
    array_0d_tag>
{

    // if(0d, 2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(result, 0));
        assert(size(true_value, 1) == size(result, 1));

        if_then_else_0d_2d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_0d_tag,
    array_2d_tag,
    array_0d_tag>
{

    // if(0d, 2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(result, 0));
        assert(size(true_value, 1) == size(result, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_0d_2d_0d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_0d_tag,
    array_0d_tag,
    array_2d_tag>
{

    // if(0d, 0d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(false_value, 0) == size(result, 0));
        assert(size(false_value, 1) == size(result, 1));

        if_then_else_0d_0d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_0d_tag,
    array_0d_tag,
    array_2d_tag>
{

    // if(0d, 0d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(false_value, 0) == size(result, 0));
        assert(size(false_value, 1) == size(result, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_0d_0d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_0d_tag,
    array_2d_tag,
    array_2d_tag>
{

    // if(0d, 2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(result, 0));
        assert(size(true_value, 1) == size(result, 1));
        assert(size(false_value, 0) == size(result, 0));
        assert(size(false_value, 1) == size(result, 1));

        if_then_else_0d_2d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_0d_tag,
    array_2d_tag,
    array_2d_tag>
{

    // if(0d, 2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(result, 0));
        assert(size(true_value, 1) == size(result, 1));
        assert(size(false_value, 0) == size(result, 0));
        assert(size(false_value, 1) == size(result, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_0d_2d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_2d_tag>
{

    // if(2d, 2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(false_value, 0) == size(condition, 0));
        assert(size(false_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_else_2d_2d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_2d_tag>
{

    // if(2d, 2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(false_value, 0) == size(condition, 0));
        assert(size(false_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_2d_2d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_0d_tag>
{

    // if(2d, 2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_else_2d_2d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_0d_tag>
{

    // if(2d, 2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_2d_2d_0d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_0d_tag,
    array_2d_tag>
{

    // if(2d, 0d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(false_value, 0) == size(condition, 0));
        assert(size(false_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_else_2d_0d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_0d_tag,
    array_2d_tag>
{

    // if(2d, 0d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(false_value, 0) == size(condition, 0));
        assert(size(false_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_2d_0d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result,
    typename ExecutionPolicy>
struct IfThenByExecutionPolicy
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        IfThenByArgumentCategory<InputNoDataPolicy, OutputNoDataPolicy,
            Condition, TrueValue, Result,
            ExecutionPolicy,
            base_class<argument_category<Condition>, array_2d_tag>,
            base_class<argument_category<TrueValue>, array_2d_tag>>
                ::apply(input_no_data_policy, output_no_data_policy,
                    execution_policy, condition, true_value, result);

    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
struct IfThenByExecutionPolicy<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    ExecutionPolicy>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                IfThenByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Condition,
                    TrueValue,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Condition>, array_2d_tag>,
                    base_class<argument_category<TrueValue>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            boost::get<SequentialExecutionPolicy>(
                                execution_policy),
                            condition, true_value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                IfThenByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Condition,
                    TrueValue,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Condition>, array_2d_tag>,
                    base_class<argument_category<TrueValue>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            boost::get<ParallelExecutionPolicy>(
                                execution_policy),
                            condition, true_value, result);
                break;
            }
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result,
    typename ExecutionPolicy>
struct IfThenElseByExecutionPolicy
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        IfThenElseByArgumentCategory<
            InputNoDataPolicy, OutputNoDataPolicy,
            Condition, TrueValue, FalseValue, Result,
            ExecutionPolicy,
            base_class<argument_category<Condition>, array_2d_tag>,
            base_class<argument_category<TrueValue>, array_2d_tag>,
            base_class<argument_category<FalseValue>, array_2d_tag>>
                ::apply(input_no_data_policy, output_no_data_policy,
                    execution_policy, condition, true_value, false_value,
                    result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
struct IfThenElseByExecutionPolicy<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ExecutionPolicy>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                IfThenElseByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Condition,
                    TrueValue,
                    FalseValue,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Condition>, array_2d_tag>,
                    base_class<argument_category<TrueValue>, array_2d_tag>,
                    base_class<argument_category<FalseValue>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            boost::get<SequentialExecutionPolicy>(
                                execution_policy),
                            condition, true_value, false_value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                IfThenElseByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Condition,
                    TrueValue,
                    FalseValue,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Condition>, array_2d_tag>,
                    base_class<argument_category<TrueValue>, array_2d_tag>,
                    base_class<argument_category<FalseValue>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            boost::get<ParallelExecutionPolicy>(
                                execution_policy),
                            condition, true_value, false_value, result);
                break;
            }
        }
    }

};

} // namespace dispatch


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    dispatch::IfThenByExecutionPolicy<InputNoDataPolicy, OutputNoDataPolicy,
        Condition, TrueValue, Result, ExecutionPolicy>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            condition, true_value, result);
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    dispatch::IfThenElseByExecutionPolicy<InputNoDataPolicy, OutputNoDataPolicy,
        Condition, TrueValue, FalseValue, Result, ExecutionPolicy>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            condition, true_value, false_value, result);
}

} // namespace detail
} // namespace if_
} // namespace algorithm
} // namespace fern
