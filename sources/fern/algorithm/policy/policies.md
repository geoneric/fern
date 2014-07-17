Policies {#fern_algorithm_policies}
========

[TOC]


Policies are used to customize algorithms with respect to:
- How to execute: [execution policy](#execution_policy).
- How to detect no-data in the inputs: [input-no-data policy](#input_no_data_policy).
- How to write no-data to the outputs: [output-no-data policy](#output_no_data_policy).
- How to determine whether input values are in the algorithm's domain: [out-of-domain policy](#out_of_domain_policy).
- How to determine whether an algorithm's result value is in the domain of the result's value type: [out-of-range policy](#out_of_range_policy).


Execution policy {#fern_algorithm_policies_execution_policy}
================
- [SequentialExecutionPolicy](@ref fern::SequentialExecutionPolicy)
- [ParallelExecutionPolicy](@ref fern::ParallelExecutionPolicy)


Input-no-data policy {#fern_algorithm_policies_input_no_data_policy}
====================
TODO


Output-no-data policy {#fern_algorithm_policies_output_no_data_policy}
=====================
TODO


Out-of-domain policy {#fern_algorithm_policies_out_of_domain_policy}
====================
An out-of-domain policy's job is to test argument values and determine whether they are valid input for the algorithm. For example, the out-of-domain policy of the square root algorithm (fern::sqrt::OutOfDomainPolicy) tests whether the argument value is not negative. Out-of-domain policies slow down an algorithm, because they need to perform tests for each value calculated. If you do not need to test for out-of-domain input values, you can use fern::DiscardDomainErrors.


~~~~{.c}
// Policy for a binary algorithm.
MyPolicy<double, double> policy;
if(policy.within_domain(value1, value2)) {
    // OK, algorithm should be able to handle argument values.
    // Call algorithm with these values.
}
else {
    // Operation can't handle these values.
    // Exception, set no-data, ...
}
~~~~

- [DiscardDomainErrors](@ref fern::DiscardDomainErrors)


Out-of-range policy {#fern_algorithm_policies_out_of_range_policy}
===================
An out-of-range policy's job is to test result values and determine whether they fit within the result's value type. For example, adding two large integral values may result is a value that is larger than the resulting integer value type. In the case of the add algorithm (fern::algebra::add), the out-of-range policy (fern::add::OutOfRangePolicy) needs the argument values as well as the result value to be able to determine whether the result is out of range. This is because when integral values go out of range, they *wrap*, eg: adding two large integral value results in a negative value.

~~~~{.c}
// Policy for a unary algorithm.
MyOutOfRangePolicy policy;
if(policy.within_range(value, result)) {
    // OK, algorithm calculated a result that fits the result's value type.
    // Use the value.
}
else {
    // Operation's result value falls outside the range of the result's
    // value type.
    // Exception, set no-data, ...
}
~~~~

- [DiscardRangeErrors](@ref fern::DiscardRangeErrors)


See also {#fern_algorithm_policies_see_also}
========
- [Wikipedia on policy-based design](http://en.wikipedia.org/wiki/Policy-based_design)
- Domain
    - [Wikipedia on arithmetic overflow](https://en.wikipedia.org/wiki/Arithmetic_overflow)
    - [Wikipedia on arithmetic underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow)
    - [Wikipedia on integer overflow](https://en.wikipedia.org/wiki/Integer_overflow)
- Range
    - [Wikipedia on range in computer science](https://en.wikipedia.org/wiki/Range_(computer_science))
    - [Wikipedia on range in mathematics](https://en.wikipedia.org/wiki/Range_(mathematics))
