.. _tutorial:

Tutorial
========
Let us assume that we simply need to divide two values. This task is very simple ofcourse, but we are going to complicate things a bit as we progress.

Looking in the Fern.Algorithm API documentation, we find the divide algorithm categorized under Algorithms | Algebra | Elementary algebra. You can also type `divide` in the search box and select `fern::algebra::divide` from the search results in the dropdown box. Or you can pick the `divide` algorithm from the list of algorithms presented in the `fern::algebra` namespace documentation page.

Each algorithm is a function template. When calling an algorithm, we must always provide the function arguments, of course. Depending on our needs, we must also provide one or more template arguments. The algorithm arguments consist of one or more input argument values and the result value. The type of these values depends on the context. Fern.Algorithm uses specific template functions (customization points) to access the values, and as long as they are implemented, the algorithm can be called. This has the advantage that the implementation of the algorithms doesn't depend on a certain type. We will see how this works later on in this tutorial.

Each ``fern::divide`` overload also accepts an execution policy argument which must be provided. This argument determines how the algorithm should perform its work. Currently we can use a ``fern::SequentialExecutionPolicy`` instance or a ``fern::ParallelExecutionPolicy`` instance.


One algorithm, multiple data types
----------------------------------
Divide two numbers
~~~~~~~~~~~~~~~~~~
Let's now just divide two floating point numbers:

.. literalinclude:: source/tutorial_divide-1.cc
   :language: cpp

Here, we explicitly include the one header that contains the definition of the ``fern::divide`` algorithm. We could also have included any of folowing headers, which would have provided us with increasingly more algorithms:

+-----------------------------------------+------------------------------------+
| header                                  | algorithms                         |
+=========================================+====================================+
| ``fern/algorithm/algebra/elementary.h`` | Elementary algebraic algorithms    |
+-----------------------------------------+------------------------------------+
| ``fern/algorithm/algebra.h``            | Algebraic algorithms               |
+-----------------------------------------+------------------------------------+
| ``fern/algorithm.h``                    | All algorithms                     |
+-----------------------------------------+------------------------------------+

The Fern.Algorithm API documentation contains information about all header files that can be included.

The example shows how to divide two ``double`` values and store the result in another ``double`` value. As you know, this is just a more complicated way of writing:

.. code-block:: cpp

   result = value1 / value2;


Divide two arrays
~~~~~~~~~~~~~~~~~
Let's complicate things a bit and assume that we need to divide an array by another array, element-wise:

.. literalinclude:: source/tutorial_divide-2.cc
   :language: cpp

Here, we changed the floating point values to arrays of floating point values. The algorithm call has not changed at all. We do have to include an additional header file: ``fern/core/vector_traits.h``. To allow Fern.Algorithm algorithms to have access to data values, while not knowing the type of the data values passed into the algorithms, it uses overloads of `getter` functions. Each algorithm assumes that two functions called `get` exist, one of which must return a writable reference to a data value and one of which must return a readable reference to a data value. That way, you can use your own data types when calling Fern.Algorithm algorithms, as long as you provide these getter functions. In the case of ``std::vector<T>``, we have already implemented these functions, and they are defined in the ``fern/core/vector_traits.h``.

.. important::

   Fern.Algorithm algorithms are function templates, not functions. The compiler uses the function templates to generate functions, given the (default) template arguments.

.. important::

   When including a header file with `getter` functions, it is crucial to do so before including the header file with the algorithm. When parsing the algorithm template's implementation, the compiler must have all information it needs to assemble an algorithm's function.

The implementation of the divide algorithm that is selected by the compiler given the arguments we passed uses functionality from the ``fern_algorithm`` library. For this example to work, we need to link it.

For 0D values (scalars), Fern.Algorithm assumes that these function templates are implemented:

.. code-block:: cpp

   template<>
   ScalarValueType const& get(Scalar const& value);

   template<>
   ScalarValueType& get(Scalar& value);

A value type is a type of an individual value. In case of ``int`` and ``std::vector<int>``, the value types are the same: ``int``.

For 1D values (arrays), Fern.Algorithm assumes that these function templates are implemented:

.. code-block:: cpp

   template<>
   ArrayValueType const& get(
       Array const& array,
       size_t index);

   template<>
   ArrayValueType get(
       Array& array,
       size_t index);


Divide a grid by a scalar
~~~~~~~~~~~~~~~~~~~~~~~~~
The previous sections show that the single ``fern::divide`` algorithm can be used for dividing different types of values. Here is another example in which we divide a 2D grid of values by a single value:

.. literalinclude:: source/tutorial_divide-3.cc
   :language: cpp

Here we use the ``fern::Array`` class template and include its `traits header file` but, apart from that, the code hasn't changed. The idea, of course, is that Fern.Algorithm provideѕ algorithms that can be used with any type of data, as long as it makes sense and the required `getter` functions are provided. If you have crafted your own collection class, then you can also use the Fern.Algorithm algorithms.


One algorithm, multiple behaviors
---------------------------------
We now know Fern.Algorithm algorithms can accept arguments with different data types and value types. Each algorithm combines these arguments to come up with a result value. The procedure at the core of each algorithm is what makes the algorithm unique. But there are aspects about each algorithm that are configurable by the caller. These aspects are called ``policies`` in C++ lingo. Currently, the algorithms accept policies that determine:

- How to execute the algorithm: sequential or concurrent: ``ExecutionPolicy``
- Whether or not to check argument values for out-of-domain errors:
  ``OutOfDomainPolicy``
- Whether or not to check result values for out-of-range errors:
  ``OutOfRangePolicy``
- Whether or not to check argument values for no-data: ``InputNoDataPolicy``
- Whether or not to write no-data to the result: ``OutputNoDataPolicy``

Which policies to use when calling an algorithm depends on the context. You need to pick policies based on your knowledge of the executation context and argument values. If an algorithm is called from within a thread of execution, you don't want the algorithm to spawn threads of it own, for example. And when processing user-profided data, you probably have to use policies that check the input for no-data and domain errors. If you know the input does not contain no-data and/or domain errors, then you are better off selecting policies that don't check for these. The resulting algorithm will be faster.

Fern.Algorithm contains a small library of often used policies. In case you have special needs then you can write your own policy classes.

Let's now return to our example of dividing two argument values, but use an algorithm that will divide the work over all CPU cores in the computer:

.. literalinclude:: source/tutorial_divide-4.cc
   :language: cpp

This example uses an instance of the fern::algorithm::ParallelExecutionPolicy. This will tell the algorithm to concurrently calculate the result.

In many real-world contexts, we need to handle no-data in the argument values. For this we can use one of the input no-data policies. The algorithm will use this policy internally to determine whether or not an element contains no-data. No-data elements can be signalled by special values (eg: -999, ``std::numerical_limits<int32_t>::min()``), or a seperate mask with boolean values or bits. This doesn't matter to the algorithm. As long as the input no-data policy provides the correct anwser.

In the next example, we copy the no-data elements of both input arguments to the result. This is often what needs to happen with no-data, but this depends on the algorithm. The advantage of doing this is that we can use a simple input no-data policy that check the result for no-data. The algorithm doesn't care, it just asks the policy whether or not the current element represents no-data or not.


.. literalinclude:: source/tutorial_divide-5.cc
   :language: cpp


One algorithm template, many instantiations
-------------------------------------------
For you as a user, the Fern.Algorithm library may seem to have a single implementation for each algorithm and this single algorithm is capable of doing its thing in many different contexts. In fact, the library contains algorithm templates, which are instantiated by the compiler. If you use a single algorithm in different contexts in a single application, you may end up with multiple instantiations of the same algorithm, all behaving slightly different.
