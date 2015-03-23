Generalization 1: data type
===========================

This requirement is the easiest to fulfill. By turning our previous
function into a function template, we can parameterize the function by
the data types of the arguments and the result.

.. literalinclude:: source/pow_2/pow.h
   :caption: pow_2/pow.h
   :language: cpp

This algorithm can be called like this:

.. literalinclude:: source/pow_2/main.cc
   :caption: pow_2/main.h
   :language: cpp

Again, some observations:

- The result is now passed as a writable argument to the function. The
  compiler instantiates a function for each unique set of
  template arguments that we provide. By turning the result into an
  argument, these instantiations are involved in overload resolution allowing
  the compiler to pick the best match. This is relevant if we call the
  algorithm for the same `Base` and `Exponent` types, but different `Result`
  types. If we had `returned` the result the compiler would have thrown
  an error complaining about ambiguous overloads. It cannot choose between
  two functions with the same argument types, but different return types.
- Although we turned the argument and result data types into template
  arguments, we are still limiting our users to the ``fern::Raster``
  interface. Only raster types adhering to this interface can be
  passed. Ideally, we want to decouple the algorithm from the data type
  interface, allowing the user to use any raster-like type, whatever its
  interface.
- Although ``Base`` is a raster-like type, ``Exponent`` must still be a
  floating point type. We would like to also be able to pass a raster-like
  type for the ``exponent`` argument.

Let's tackle these limitations in turn. A technique to decouple an algorithm
from its argument's interfaces is to use customization points. And we can
use a technique called tag dispatching to select an algorithm's
implementation based on properties of the template's argument types


Customization points
--------------------
Let us forget about a single specific class for representing rasters and focus
on the minimal set of requirements that the algorithm poses on the arguments
passed in. Given that the algorithm currently takes a raster as its first
argument and a number as its second, we can define the folowing requirements:

- It must be possible to determine the number of rows and columns in the
  ``base`` argument raster.
- It must be possible to iterate over all raster cells and obtain a readable
  value of each cell from the ``base`` argument raster and a writable
  reference of each cell in the ``result`` raster.

Customization points are free function templates that return a property
of an argument and which can be overloaded for different argument types. One
example from the C++ Standard Library is `std::begin`_, which returns an
iterator to the first element in a collection. This function is overloaded
for the various containers in the STL and for C-style arrays. Using
``std::begin`` to obtain such an iterator is a better option than calling
``my_container.begin()`` because the latter assumes ``my_container``'s
type provides this function. C-style arrays don't, for example.

Here is a declaration of a customization point for getting the size of a
raster:

.. literalinclude:: source/pow_3/customization_point.h
   :caption: pow_3/customization_point.h: size
   :language: cpp
   :lines: 5-7

It takes a raster and an id of a dimension. In the case of 2D rasters,
this latter argument must be a 0 or a 1, where 0 represents the first
dimension in a 2D array and 1 the second. Since C-style arrays have a
`row-major ordering <http://en.wikipedia.org/wiki/Row-major_order>`_,
the first dimension iterates over the rows and the second over the
columns.

We can implement this customization point for our ``Raster`` class template
as folows, using a `partial specialization`_:

.. literalinclude:: source/pow_3/customization_point/raster.h
   :caption: pow_3/customization_point/raster.h: size
   :language: cpp
   :lines: 6-14

For raster types with other interfaces we can implement alternative
overloads. In our algorithm we can now replace the code obtaining the size
of the dimensions to be calls to our ``size`` customization point.

We need two more customization points: one for obtaining a readable value
of a specific cell and one for obtaining a writable reference to a value
of a specific cell.

.. literalinclude:: source/pow_3/customization_point.h
   :caption: pow_3/customization_point.h: get
   :language: cpp
   :lines: 9-15

The return type is calculated by the ``value_type`` `alias template`_, which
returns the value type of the template argument. Its definition assumes
that type traits (see below) provide the value type of every data type:

.. literalinclude:: source/pow_3/type_traits.h
   :caption: pow_3/type_traits.h: value_type
   :language: cpp
   :lines: 11-13

For our ``Raster`` template class we can implement the additional
customization points as folows:

.. literalinclude:: source/pow_3/customization_point/raster.h
   :caption: pow_3/customization_point/raster.h: get
   :language: cpp
   :lines: 17-37

The last missing piece is the implementation of the TypeTraits class
template. Type traits provide properties of a type. In our case, we need
the value type of the ``Raster`` template class.

.. literalinclude:: source/pow_3/type_traits.h
   :caption: pow_3/type_traits.h
   :language: cpp
   :lines: 4-8

.. literalinclude:: source/pow_3/type_traits/raster.h
   :caption: pow_3/type_traits/raster.h
   :language: cpp
   :lines: 5-12

Let's now rewrite our algorithm in terms of customization points.

.. literalinclude:: source/pow_3/pow.h
   :caption: pow_3/pow.h
   :language: cpp

Note that the algorithm is written in terms of template arguments and
customization points, all to be provided by the caller. We abstracted
away any reference to a specific raster type. With this in place, we
can now call this algorithm with every raster-like base argument for
which we have implemented the three customization points. In the next
section we are going to get rid of the limitation that the base argument
must be a raster and the exponent a number.


Tag dispatching
---------------
If we call our current ``pow`` algorithm with a ``Raster`` instance as
the exponent argument, the compilation fails. Our algorithm expects the
exponent to be a single number. Passing a ``Raster`` requires a different
implementation. The problem we need to solve is how we can pass different
permutations of raster and number arguments to ``pow``, without having to
revert to differently named functions like ``pow_raster_number`` and
``pow_raster_raster``. Somehow, our ``pow`` implementation needs to do
something different depending on its template argument types.

`Tag dispatching`_ can help us here. What we need is a tag for each data
type we support and call a second function that actually implements
the algorithm for the specific combination of argument types. Let us first
define the tags:

.. literalinclude:: source/pow_4/type_traits.h
   :caption: pow_4/type_traits.h, tags
   :language: cpp
   :lines: 4-5

Now we need a way to obtain the tag given a data type. We can use our traits
class for that:

.. literalinclude:: source/pow_4/type_traits.h
   :caption: pow_4/type_traits.h, tag type traits
   :language: cpp
   :lines: 8-18

.. literalinclude:: source/pow_4/type_traits/raster.h
   :caption: pow_4/type_traits/raster.h, tag type traits
   :language: cpp
   :lines: 5-13

In our original ``pow`` implementation we can now dispatch on data type tag
to an algorithm with the correct implementation.

.. literalinclude:: source/pow_4/pow.h
   :caption: pow_4/pow.h
   :language: cpp

More overloads of ``detail::pow`` can be implemented if necessary.

With these changes we successfuly solved all limitations identified in
the beginning of this section. In the next section we are going to take
a look at handling invalid data elements.


.. _partial specialization: http://en.cppreference.com/w/cpp/language/partial_specialization
.. _alias template: http://en.cppreference.com/w/cpp/language/type_alias
.. _Tag dispatching: http://www.boost.org/community/generic_programming.html#tag_dispatching
.. _std::begin: http://en.cppreference.com/w/cpp/iterator/begin

