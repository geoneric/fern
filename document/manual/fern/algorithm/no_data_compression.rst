.. _no_data_compression:

No-data compression
===================
No-data compression is about creating a copy of a collection of values, but without the no-data elements. The goal is to end up with a smaller collection that can be passed to algorithms that don't need to test for no-data. Under certain circumstances, this will speed up the calculations considerably. Unfortunately, there are some drawbacks too. In this section we will provide some guidance as to when no-data compression might be useful and when it might not be.

You might want to consider compressing your collections when all of these conditions are true:

- Your data contains no-data elements. If your data contains no no-data elements, you should use an input-no-data policy that skips the no-data test, instead of creating a compressed collection. Otherwise you will suffer from the overhead involved with no-data compression, without profiting from the benefits.
- You want to call at least one algoritm that does not care about the relative location of individual values in the value collection. In a compressed collection, values end up in different locations than in the original collection they are copied from. The more of these algorithms you have to call, the more you will benefit from no-data compression.
- All collections that need to be compressed have no-data elements at the same location. Compressing such collections will result in compressed collections of the same size.
- The algorithms you want to call do not generate no-data in the result collections. There is no way to mark no-data in the result collection, so you have to be certain that there is no reason for the algorithms to generate them. In practice this means that for each algorithm you want to call with compressed collections:
    - All argument values are within the domain of the algorithm.
    - All result values are within the range of the result's value type.

No-data compression involves these steps:

#. Compress argument collections to compressed collections.
#. Call algorithms with compressed collections. Pass policies to the algorithms that:
    - Do not test for no-data in the argument collections.
    - Do not test for out-of-domain values in the argument collections.
    - Do not test for out-of-range values in the result collections.
#. Decompress the compressed result collections you need to process further.

As you see, there are some steps involved that are unique to no-data compression: compressing the argument collections and decompressing the result collections. These steps take time and may well mean that no-data compression is not useful. The only way to tell so is by measuring the efficiency of the resulting code. It is easy to time a set of algorithms using uncompressed collections and using compressed collections and compare the runtimes.

The relative benefit of using no-data compression is affected by:

- The size of the data collections. There may be a threshold in the number of values in the collections below which the overhead of no-data compression dominates the possitive effect.
- The execution policy passed to the algorithms.
- The hardware available to the calculations.

The only way to tell whether no-data compression is useful or not is by timing the code. Fern contains the Stopwatch class that is useful for timing snippets of code.
