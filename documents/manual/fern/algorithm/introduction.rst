.. _introduction:

Introduction
============
The Fern.Algorithm C++ library is part of the Fern modeling software, built by `Geoneric <http://www.geoneric.eu>`_. The implemented algorithms translate one or more input argument values and into a result argument value. A single value can be a scalar value (0D), an array of values (1D), a 2D array of values, or another type of value. This depends on what makes sense for each algorithm. For example, Fern.Algorithm contains the ``add`` algorithm for adding two values and writing the result to a result value. A unique property of the implementation is that `policies <http://en.wikipedia.org/wiki/Policy-based_design>`_ are used to make it possible to configure various aspects of each algorithm's behavior. This makes the algorithms usable in many different contexts, but without sacrificing performance.

When writing an algorithm without knowing anything about the context in which the algorithm is used, it is important to make the aspects of the algorithm that depend on the context configurable. The one and only unique aspect of the algorithm is the rule for translating input values to a result value. For example, the unique aspect of the ``sqrt`` algorithm is that it calculates the square root of its input argument value. Every other algorithm calculates something else. This is the one and only aspect of an algorithm that is not configurable by the caller.

But an algorithm that is useful in different contexts must handle other algorithm aspects too. Depending on the data used and the individual algorithm, it may be necessary to skip no-data values, detect out-of-domain values, and/or detect out-of-range values. And if so, it most probably is necessary to write no-data values in these cases. But it is often crucial that it isn't necessary to pay for something that is not used. In case it is known that the input value does not contain no-data, then the algorithm should not spend time checking for it. Also, the caller may want algorithms to use more than a single CPU core, but maybe not always.

All these aspects can be configured when using Fern.Algorithm. As a developer, each time you call a generic Fern.Algorithm algorithm with a specific set of policies, you configure an algorithm. At compile-time, this algorithm gets assembled into a runtime algorithm by the compiler. Depending on the compiler options, the compiler is able to optimize the configured algorithm into machine code that is as efficient as would a corresponding function be that was specifically crafted by a developer.

Concluding, Fern.Algorithm algorithms are written in terms of policies that can be provided by the developer calling them. That way he gets exactly the algorithm he needs, without sacrificing performance. In the :ref:`tutorial`, we will show you how this works.


