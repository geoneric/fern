.. _abstraction_no_data:

Abstraction 2: no-data
======================
The algorithm we developed until now assumes all argument raster
cells contain valid data. As mentioned in the :ref:`introduction
<introduction>`, this is often not the case. Ideally, we want to be able
to tell the algorithm whether or not our argument rasters may contain
invalid cells or not, and if so, how to test each individual cell for
validity.

Similarly, instead of being able to *test argument raster cells* for
validity, we want to be able to *mark result raster cells* as invalid,
for example in case one of the argument rasters contains invalid cells
or contains values that are considered out of the algorithm's domain
(see :ref:`abstraction_domain_range`).

`Policy classes`_ are a technique that can help us with these
requirements. They allow us to parameterize *behavior* of an algorithm,
which enables the caller to tune the algorithm to her needs. An example
of a policy from the C++ Standard Library is `Allocator`_, which allows
the caller to tune the allocation and deallocation of memory in code
accepting an allocator policy (most containers from the Standard Library
do, see for example the second template argument of the `std::vector`_
class template).

Our algorithm needs to be parameterized with policy classes
that encapsulate the behavior of how to deal with no-data values in the
argument and result rasters. We must rewrite our algorithm to use the
no-data testing policy class for testing individual raster elements. The
caller determines if and how these tests are performed.  Depending on
the raster type used, no-data may be marked by a special marker value,
like the maximum or minimum representable value by the value type, or
maybe by a layered boolean raster or a bit mask. It all depends on the
type passed in the algorithm. As long as the algorithm has a way to test
and mark no-data elements, it can do its job.

Enough theory, lets think about the interface of the no-data checking and
marking policies. For ease of reference we call the no-data checking policies input no-data policies, and the no-data marking policies output no-data policies.
Both kinds of no-data policies need to be able to either test or mark individual elements for validity. In the current algorithm, individual elements are addressed using row and column indices, which we can use for our no-data policies too.
So, a input no-data policy needs to be able to tell whether or not an element with a certain row and column index contains valid data or not:


.. code-block:: c++

   bool MyInputNoDataPolicy::is_no_data(size_t row, size_ col);


Similarly, an output no-data policy needs to be able to mark an element as valid or invalid:


.. code-block:: c++

   bool MyOutputNoDataPolicy::mark_no_data(size_t row, size_ col);


We are not passing the raster itself, because we have no idea how
the policy is implemented. It may use the raster, or not. The policy
instance must be self-contained and be able to answer our requests,
only given an element's address.

In general, algorithms may accept zero or more arguments, which all may
have a different way to mark no-data cells (if at all). So, instead of
a single input no-data policy we need an input no-data policy for every
input argument. And in the algorithms we need to test argument elements
using the appropriate policy.

Generalizing further, being able to test and mark validity is as relevant
for simple numbers as for rasters containing numbers. For example, when
calculating the sum of the values in a raster, the result may be larger
than the type used to represent the result can hold (out of range error,
see :ref:`abstraction_domain_range`. In that case, the caller may want
the algorithm to mark the result as being invalid.

So, in the algorithm, we likely use code like this (assuming a raster
``base`` and number ``exponent``:


.. code-block:: c++

   if(std::get<0>(input_no_data_policy).is_no_data(r, c) ||
           std::get<1>(input_no_data_policy).is_no_data()) {
       output_no_data_policy.mark_no_data(r, c);
   }
   else {
       get(result, r, c) = std::pow(get(base, r, c), exponent);
   }


In words: if the current ``base`` raster cell is not valid or if the exponent is not valid, then mark the current ``result`` raster cell as not valid. In all other cases, calculate a result given the valid argument values.

It may seem that this adds quite some code to our simple algorithm. Aren't we adding logic for worst-case scenarios that may not be applicable to many use cases? For example, the exponent number may be a constant hardcoded by the caller, like a ``2`` for squaring the ``base``. This exponent will never be invalid. For those circumstances it is useful to have `dummy` policies that contain code that an optimizing compiler will remove from the instantiated algorithm during `code generation`_.




..
   If the argument rasters do not handle invalid cells, we do not want our algorithm to spend CPU cycles testing them.

   In case the raster doesn't have support for representing invalid elements, a dummy policy class can be implemented that does not affect the algorithm's efficiency.


.. policy class. Its sole 






.. _Policy classes: http://www.boost.org/community/generic_programming.html#policy
.. _Allocator: http://en.cppreference.com/w/cpp/concept/Allocator
.. _std::vector: http://en.cppreference.com/w/cpp/container/vector
.. _code generation: http://en.wikipedia.org/wiki/Code_generation_%28compiler%29
