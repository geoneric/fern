Policies {#policies}
========

[TOC]


Policies are used to customize algorithms with respect to:
- How to detect no-data in the inputs: [input-no-data policy](#input_no_data_policy).
- How to write no-data to the outputs: [output-no-data policy](#output_no_data_policy).
- How to determine whether input values are in the algorithm's domain: [out-of-domain policy](#out_of_domain_policy).
- How to determine whether an algorithm's result value is in the domain of the result's value type: [out-of-range policy](#out_of_range_policy).


Input-no-data policy {#input_no_data_policy}
====================
TODO


Output-no-data policy {#output_no_data_policy}
=====================
TODO


Out-of-domain policy {#out_of_domain_policy}
====================
An out-of-domain policy's job is to test argument values and determine whether they are valid input for the algorithm.

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

For example, the out-of-domain policy of the square root algorithm tests whether the argument value is not negative.

- [DiscardDomainErrors](@ref fern::DiscardDomainErrors)


Out-of-range policy {#out_of_range_policy}
===================
An out-of-range policy's job is to test result values and determine whether they fit within the result's value type.

~~~~{.c}
// Policy for a unary algorithm.
MyOutOfRangePolicy policy;
if(policy.within_range(value)) {
    // OK, algorithm calculated a result that fits the result's value type.
    // Use the value.
}
else {
    // Operation's result value falls outside the range of the result's
    // value type.
    // Exception, set no-data, ...
}
~~~~

For example, adding two large integral values may result is a value that is larger than the resulting integer value type. In the case of the add algorithm, the out-of-range policy needs the argument values as well as the result value to be able to determine whether the result is out of range. This is because when integral values go out of range, they *wrap*, eg: adding two large integral value results in a negative value.

- [DiscardRangeErrors](@ref fern::DiscardRangeErrors)


See also {#see_also}
========
- [Wikipedia on policy-based design](http://en.wikipedia.org/wiki/Policy-based_design)
- Domain
    - [Wikipedia on arithmetic overflow](https://en.wikipedia.org/wiki/Arithmetic_overflow)
    - [Wikipedia on arithmetic underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow)
    - [Wikipedia on integer overflow](https://en.wikipedia.org/wiki/Integer_overflow)
- Range
    - [Wikipedia on range in computer science](https://en.wikipedia.org/wiki/Range_(computer_science))
    - [Wikipedia on range in mathematics](https://en.wikipedia.org/wiki/Range_(mathematics))
