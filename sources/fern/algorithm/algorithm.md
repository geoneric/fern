Algorithms {#fern_algorithm}
==========

The Fern algorithms have been classified in various categories.
Each algorithm is configurable using policies. Most/all algorithms have
overloads that allow them to be called with default policies. Default policies
are always those policies that do not do anything. For example, the default
policy for testing input value for no-data is the fern::SkipNoData policy,
which does not incur any overhead.

See the folowing pages for more information:

- Categories

    - @ref fern_algorithm_algebra
    - @ref fern_algorithm_convolution
    - @ref fern_algorithm_core
    - @ref spatial_algorithms
    - @ref statistics
    - @ref trigonometry

- Support code

    - @ref fern_algorithm_policies
