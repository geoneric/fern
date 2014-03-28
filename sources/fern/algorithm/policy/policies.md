Policies {#policies}
========

Out-of-domain policy
--------------------
An out-of-domain policy's job is to test argument values and determine whether they are valid input for the operation.

~~~~{.c}
// Policy for a binary operation.
MyPolicy<double, double> policy;
if(policy.within_domain(value1, value2)) {
    // OK, operation should be able to handle argument values.
    // Call operation with these values.
}
else {
    // Operation can't handle these values.
    // Exception, set no-data, ...
}
~~~~

For example, the out-of-domain policy of the square root operation tests whether the argument value is not negative.

- [DiscardDomainErrors](@ref fern::DiscardDomainErrors)


Out-of-range policy
-------------------
An out-of-range policy's job is to test result values and determine whether they fit within the result's value type.

~~~~{.c}
// Policy for a unary operation.
MyOutOfRangePolicy policy;
if(policy.within_range(value)) {
    // OK, operation calculated a result that fits the result's value type.
    // Use the value.
}
else {
    // Operation's result value falls outside the range of the result's
    // value type.
    // Exception, set no-data, ...
}
~~~~

For example, adding two large integral values may result is a value that is larger than the resulting integer value type. In the case of the add operation, the out-of-range policy needs the argument values as well as the result value to be able to determine whether the result is out of range. This is because when integral values go out of range, they *wrap*, eg: adding two large integral value results in a negative value.

- [DiscardRangeErrors](@ref fern::DiscardRangeErrors)
