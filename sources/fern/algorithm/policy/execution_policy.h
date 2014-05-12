#pragma once
#include "fern/algorithm/policy/parallel_execution_policy.h"
#include "fern/algorithm/policy/sequential_execution_policy.h"


namespace fern {

class SequentialExecutionPolicy;


class ParallelExecutionPolicy;


constexpr SequentialExecutionPolicy seq = SequentialExecutionPolicy();

constexpr ParallelExecutionPolicy par = ParallelExecutionPolicy();

} // namespace fern
