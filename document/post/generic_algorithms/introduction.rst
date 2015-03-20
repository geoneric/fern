Generic algorithms and spatial fields
=====================================

Generic algorithms
------------------
Generic algorithms in C++ are templated algorithms that are written
in terms of template argument types that are defined later, when the
algorithms are instantiated. The `C++ Standard Template Library (STL)
<http://en.wikipedia.org/wiki/Standard_Template_Library>`_
constains many examples of generic algorithms.

To appreciate the benefits of generic algorithms over non-generic
algorithms, we can first take a look at a non-generic algorithm and
compare it to a generic one. Let's create an algorithm for finding the
maximum element in a collection:

.. literalinclude:: source/non_generic_max.cc
   :language: cpp

The algorithm accepts iterators pointing to the first and one-past-last
element in the collection. It returns an iterator pointing to the first
element with the greatest value, or last if the collection is empty.

This algorithm works fine, but only if we want to find the maximum
element in a vector of integers. Often, this is not the case, and we end
up writing multiple algorithms for multiple argument types. That is the
kind of work computers are better at than us. Fortunately, we can ask the
compiler to generate such a family of algorithms for us, by using
function templates. The generic version of ``max_element`` is part of the
STL and has the folowing prototype (we skip the overload accepting a
comparison function):

.. code-block:: cpp

   template<
       typename ForwardIt>
   ForwardIt max_element(ForwardIt first, ForwardIt last);

When we call this generic algorithm like this:

.. code-block:: cpp

   std::vector<int> my_values{1, 5, 2, 4, 3};
   auto it = std::max_element(my_values.begin(), my_values.end())

the compiler will instantiate the same algorithm for us as the non-generic
version which we wrote above. But the generic algorithm can also be
called with other argument types:

.. code-block:: cpp

   std::list<std::string> my_labels{"amsterdam", "rotterdam", "the hague"};
   auto it = std::max_element(my_labels.begin(), my_labels.end())

So, in short, generic algorithms are templates for actual algorithms,
with some parts provided by the caller. The compiler will substitute the
missing parts (the template arguments) with the information provided at
the call site and instantiate and compile the final algorithm. Using
the generic algorithm in a different context involves calling it with
different parameter types.

A generic algorithm cannot be compiled without the caller providing
essential information that the algorithm is parameterized on.

The underlying reason for developing generic algorithms is that the person
developing the algorithms doesn't have all the pieces of information
required to be able to implement the final algorithm. An important task
when designing generic algorithms is to figure out what the pieces of
information are that will vary, and what the pieces of information are
the will not vary. The first category will end up as template arguments
of the generic algorithms.


Spatial fields
--------------
To limit ourselves a bit to data that we know most about we will now
turn to spatial fields. The term field is overloaded and has many
different meanings. In our case we use the term in the geographical
sense. A field is a quantity that (at least in theory) has a value at
any point in space. This value varies continuously through space. Here
we limit ourselves to 2D fields. Examples of such fields are the
atmospheric pressure at a certain height above the earth's surface,
and the elevation of the earth's surface itself. To be able to handle
such quantities in the computer, we discretize fields using rasters,
which are essentially 2D arrays with some additional information about
where each raster is positioned according to some coordinate reference
system. For our purpose here it is enough to think of spatial fields as
2D arrays positioned somewhere.

In the next section we will develop a first version of a simple
algorithm handling a 2D spatial field. We we will generalize it a bit,
but we will also see that it is still quite limited.
