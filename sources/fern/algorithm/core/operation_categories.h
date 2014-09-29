#pragma once


namespace fern {
namespace algorithm {

// Operation categories. Used in tag dispatching.

//! Local operation only use the local value(s) to determine the output value.
/*!
  Local operations don't need access to surrounding values (in time and/or
  space) to calculate the result value.
*/
struct local_operation_tag {};

//! Local operation only use the local value(s) to determine the output value.
/*!
  Local operations don't need access to surrounding values (in time and/or
  space) to calculate the result value.

  Aggregate operations combine input values to values with less dimensions. An
  example of this is the count operation that determines the number of values
  in the input that is equal to some other value.
*/
struct local_aggregate_operation_tag: local_operation_tag {};

//! Focal operations use the local value(s) and the surrounding values to determine the output value.
/*!
*/
struct focal_operation_tag {};

struct zonal_operation_tag {};

struct global_operation_tag {};

} // namespace algorithm
} // namespace fern
