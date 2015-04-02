.. _introduction:

Introduction
============

Generic algorithms
------------------
Generic algorithms in C++ are templated algorithms that are written
in terms of template argument types that are defined later, when the
algorithms are instantiated. The `C++ Standard Template Library (STL)`_
constains many examples of generic algorithms.

To appreciate the benefits of generic algorithms over non-generic
algorithms, we can first take a look at a non-generic algorithm and
compare it to a generic one. Let's define an algorithm for finding the
maximum element in a collection:

.. literalinclude:: source/max_element/max_element.cc
   :language: cpp

The algorithm accepts iterators pointing to the first element and end
position of the collection. It returns an iterator pointing to the first
element with the greatest value, or the end position if the collection
is empty.

This algorithm works fine, but only if we want to find the maximum
element in a vector of integers. When this is not the case, we end
up writing multiple algorithms for multiple argument types. That is the
kind of work computers are better at than we. Fortunately, we can ask the
compiler to generate such a family of algorithms for us, by using
function templates. The generic version of ``max_element`` is part of the
STL (`std::max_element`_) and has the folowing prototype (we skip the overload accepting a
user-defined comparison function):

.. code-block:: cpp

   template<
       typename ForwardIt>
   ForwardIt max_element(ForwardIt first, ForwardIt last);

When we call this generic algorithm like this:

.. code-block:: cpp

   std::vector<int> my_values{1, 5, 2, 4, 3};
   auto it = std::max_element(my_values.begin(), my_values.end());

the compiler will instantiate a similar algorithm for us as the non-generic
version which we wrote above. But the generic algorithm can also be
called with other argument types:

.. code-block:: cpp

   std::list<std::string> my_labels{"amsterdam", "rotterdam", "the hague"};
   auto it = std::max_element(my_labels.begin(), my_labels.end());

So, in short, generic algorithms are templates for actual algorithms,
with some aspects provided by the caller. The compiler will substitute the
missing aspects (the template arguments) with the information provided at
the call site and instantiate and compile the final algorithm. Using
the generic algorithm in a different context involves calling it with
different parameter types.

A generic algorithm cannot be compiled without the caller providing
essential information that the algorithm is parameterized on.

One reason for developing generic algorithms is that the person
developing the algorithms doesn't have all the pieces of information
required to be able to implement the final algorithm. An important task
when designing generic algorithms is to figure out what the pieces of
information are that users will want to vary. These will end up as
template arguments of the generic algorithms.


Spatial fields
--------------
To limit ourselves a bit to data that we know most about we will now
turn to spatial fields. The term field is overloaded and has many
different meanings. In our case we use the term in the spatial
sense. A field is a quantity that (at least in theory) has a value at
any point in space. This value varies continuously through space. Here
we limit ourselves to 2D fields. Examples of such fields are the
atmospheric pressure at a certain height above the earth's surface,
and the elevation of the earth's surface itself (both projected on a
2D carthesian plane). To be able to conveniently handle such quantities in the
computer, we discretize fields using rasters, which are essentially
2D arrays with some additional information about where each raster is
positioned according to some coordinate reference system, and the width
and height of the raster cells. For our purposes here it is enough to
think of spatial fields as 2D arrays positioned somewhere.

.. image:: image/field_raster.*

Next, we will turn to the requirements of our generic algorithms.


Requirements
------------
In this section we will develop a first version of a generic algorithm
handling 2D spatial field data. It is good practice to start with a
non-generic version, and get a clear idea about the shortcomings, before
writing generic code.

For our purposes here we are not interested in the calculations
themselves. In fact we are interested in everything but the calculations
themselves. The algorithm we will develop will simply raise each cell in
a raster to the power of some exponent. Here is a first try:

.. literalinclude:: source/pow_1/pow.cc
   :language: cpp

We use the ``Raster`` class template from `Geoneric`_'s Fern library, but
this could be any class with which we can model a raster. The details
of this class' implementation don't matter here.

Some observations and drawbacks of this implementation:

- The type of the individual elements (further on we call this the `value
  type`) in the ``base`` raster are fixed to ``double``. This is limiting the
  caller. She might prefer to pass in ``float`` or ``long double``
  values. The same is true for ``exponent``.
- The type of the ``base`` argument (the `data type`) is fixed to
  ``Raster<double, 2>``. This is also limiting the caller. She has to
  use our ``Raster`` type while she may have a type for modelling rasters
  that fits her needs better.
- The exponent is fixed to a ``double``, but we may want to be able to
  pass a raster instead of a single number. When we pass a raster for
  ``exponent``, we expect each cell in ``base`` to be raised by the
  corresponding cell in ``exponent``.
- The result is returned, which is OK if ``Raster`` supports move semantics.
  Otherwise it is definitely not OK. We don't want to copy possibly very
  large rasters. Furthermore, in C++ return types are
  not considered in overload resolution which limits our possibilities to
  make the return type configurable by the caller. This will be explained
  in more detail later on.
- Often, algorithms have a limited domain of valid input values. For
  example, a negative finite base and a non-integer finite exponent are
  considered out of domain by the ``std::pow`` algorithm. What should
  happen if an algorithm's argument(s) contains values that are not part of
  the algorithm's valid domain of input values?
- Similarly, result value types have a limited range of values they can
  represent. For example, an ``int8_t`` cannot hold values larger than
  :math:`(2^8 / 2) - 1`. What should happen if an algorithm calculates
  values that are not part of the valid range of values?
- In the algorithm above, we perform a calculation for each cell. In reality,
  rasters often contain cells with undefined values. Depending on the field
  that is represented by the raster, this may be due to clouds,
  measurement errors, errors in previous calculations, etc. We clearly
  need a way to test whether or not a cell from a raster argument contains
  a valid value or not.
- Given the previous observations, we also need a way to mark that a result
  raster cell does not contain a valid value. If an argument raster cell
  contains an invalid value we want to be able to mark the corresponding
  cell in the result raster as invalid too. Also, if we are able to detect
  out-of-domain and out-of-range errors, we want to be able to mark the
  corresponding result raster cells as invalid.

Given the non-generic algorithm and the observations, we are now better
equiped to list the requirements of a generic algorithm handling spatial
fields:

- It must be possible for the user to pick the data type and value type of
  the argument(s) and result(s). The data type does not need to be picked
  from a fixed set of types. The user must be able to use her own types.
- It must be possible for the user to 'explain' to the algorithm how to:

      - detect invalid argument raster cell values
      - mark invalid result raster cell values
      - handle out of domain argument values
      - handle out of range result values

- However the above requirements are met, the performance of the algorithm
  must be equal to a handwritten non-generic algorithm.

In the next sections, we are going to tackle each of these requirements
in turn.


.. _C++ Standard Template Library (STL): http://en.wikipedia.org/wiki/Standard_Template_Library
.. _std::max_element: http://en.cppreference.com/w/cpp/algorithm/max_element
.. _Geoneric: http://www.geoneric.eu
