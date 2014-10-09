.. _tutorial:

Tutorial
========
Let us assume that we simply need to divide two values. This task is very simple ofcourse, but we are going to complicate things a bit as we progress.

Looking in the Fern.Algorithm API documentation, we find the divide algorithm categorized under Algorithms | Algebra | Elementary algebra. You can also type `divide` in the search box and select `fern::algebra::divide` from the search results in the dropdown box. Or you can pick the `divide` algorithm from the list of algorithms presented in the `fern::algebra` namespace documentation page.

Each algorithm is a function template. When calling an algorithm, we must always provide the function arguments, of course. Depending on our needs, we must also provide one or more template arguments. The algorithm arguments consist of one or more input argument values and the result value. The type of these values depends on the context. Fern.Algorithm uses specific template functions to access the values, and as long as they are implemented, the algorithm can be called. This has the advantage that the implementation of the algorithms doesn't depend on a certain type. We will see how this works later on in this tutorial.

Each ``fern::divide`` overload also accepts an execution policy argument which must be provided. This argument determines how the algorithm should perform its work. Currently we can use a ``fern::SequentialExecutionPolicy`` instance or a ``fern::ParallelExecutionPolicy`` instance. These instances are predefined in the library, and named ``fern::sequential`` and ``fern::parallel``.


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

Here, we changed the floating point values to arrays of floating point values. The algorithm call has not change at all. We do have to include an additional header file: ``fern/core/vector_traits.h``. To allow Fern.Algorithm algorithms to have access to data values, while not knowing the type of the data values passed into the algorithms, it uses overloads of `getter` functions. Each algorithm assumes that two functions called `get` exist, one of which must return a writable reference to a data value and one of which must return a readable reference to a data value. That way, you can use your own data types when calling Fern.Algorithm algorithms, as long as you provide these ``getter`` functions. In the case of ``std::vector<T>``, we have already implemented these functions, and they are defined in the ``fern/core/vector_traits.h``.

.. important::

   Fern.Algorithm algorithms are function templates, not functions. The compiler uses the function templates to generate functions, given the (default) template arguments.

.. important::

   When including a header file with `getter` functions, it is crucial to do so before including the header file with the algorithm. When parsing the algorithm template's implementation, the compiler must have all information it needs to assemble an algorithm's function.

The implementation of the divide algorithm that is selected by the compiler given the arguments we passed uses functionality from the ``fern_algorithm_core`` library. For this example to work, we need to link it.

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
TODO


One algorithm template, many instantiations
-------------------------------------------
For you as a user, the Fern.Algorithm library may seem to have a single implementation for each algorithm and this single algorithm is capable of doing its thing in many different contexts. In fact, the library contains algorithm templates, which are instantiated by the compiler. If you use a single algorithm in different contexts in a single application, you may end up with multiple instantiations of the same algorithm, all behaving differently.

