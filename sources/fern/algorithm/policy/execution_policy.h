#pragma once


namespace fern {

class SequentialExecutionPolicy{};


class ParallelExecutionPolicy{};


constexpr SequentialExecutionPolicy sequential = SequentialExecutionPolicy();

constexpr ParallelExecutionPolicy parallel = ParallelExecutionPolicy();

} // namespace fern
