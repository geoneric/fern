Generalization 1: data type
===========================

.. note::

   Raster should be a template from the start.

This requirement is the easiest to fulfill. By turning our previous
function into a function template, we can parameterize the function by
the data types of the arguments and the result.


.. code-block:: cpp

   template<
       typename Base,
       typename Exponent,
       typename Result>
   void pow(
       Base const& base,
       Exponent const& exponent,
       Result& result)
   {
       Size const* shape{base.shape()};

       for(Index i = 0; i < shape[0]; ++i) {
           for(Index j = 0; j < shape[1]; ++j) {
               result[i][j] = std::pow(base[i][j], exponent);
           }
       }

       return result;
   }


This algorithm can be called like this:


.. code-block:: cpp

   Raster base(nr_rows, nr_cols);
   // Assign values to the raster.
   // ...
   double exponent{2};
   Raster result(nr_rows, nr_cols);

   pow(base, exponent, result);


Again, some observations:

- The result is now passed as a writable argument to the function. The
  compiler instantiates a function for each unique set of
  template arguments that we provide. By turning the result into an
  argument, these instantiations are involved in overload resolution allowing
  the compiler to pick the best match. If we had `returned` the result
  the compiler would have thrown an error complaining about ambiguous
  overloads.
- Although we turned the argument and result data types into template
  arguments, we are still limiting our users to the ``boost::multi_array``
  interface. Only base raster types adhering to this interface can be
  passed. Ideally, we want to decouple the algorithm from the data type
  interface, allowing the user to use any raster-like type.
- Although ``Base`` is a raster-like type, ``Exponent`` must still be a
  floating point type. We would like to also be able to pass a raster-like
  type for the ``exponent`` argument.

Let's tackle the limitations in turn. A technique to decouple an algorithm
from the argument's interface is to use customization points. We can use
tag dispatching to select an algorithm's implementation based on properties
of the template types


Customization points
--------------------
Let us forget about one specific class for representing rasters and focus
on the minimal set of requirements that the algorithm poses on the arguments
passed in. Given that the algorithm currently takes a raster as its first
argument and a number as its second, we can define the folowing requirements:

- It must be possible to determine the number of rows and columns in the
  base raster.
- It must be possible to iterate over all raster cells and obtain a readable
  value of each cell from the base raster and a writable reference of each cell
  in the result raster.

Customization points are free function templates that return a property
of an argument and which can be overloaded for different argument types. One
example from the C++ Standard Library is ``std::begin``, which returns an
iterator to the first element in a collection. This function is overloaded
for the various containers in the STL and for C-style arrays. Using
``std::begin`` to obtain such an iterator is a better option than calling
``my_container.begin()`` because the latter assumes ``my_container``'s
type provides this function. C-style arrays don't, for example.

Here is a declaration of a customization point for getting the size of a
raster:


.. code-block:: cpp

   template<
       typename Raster>
   size_t size(Raster const& raster, size_t dimension);


It takes a raster and an id of a dimension. In the case of 2D rasters,
this latter argument must be a 0 or a 1, where 0 represents the first
dimension in a 2D array and 1 the second. Since C-style arrays have a
`row-major ordering <http://en.wikipedia.org/wiki/Row-major_order>`_,
the first dimension iterates over the rows and the second over the
columns.

We can implement this customization point for our ``Raster`` type as folows:


.. code-block:: cpp

   template<>
   size_t size(
       Raster const& raster,
       size_t dimension)
   {
       return raster.shape()[dimension];
   }


For raster types with other interfaces we can implement alternative
overloads. In our algorithm we can now replace the code obtaining the size
of the dimensions be calls to our ``size`` customization point.

We need to more customization points. One for obtaining a readable value
of a specific cell and one for obtaining a writable reference to a value
of a specific cell.


.. code-block:: cpp

   template<
       typename Raster>
   value_type<Raster> const& get(Raster const& raster, size_t row, size_t col);

   template<
       typename Raster>
   value_type<Raster>& get(Raster& raster, size_t row, size_t col);


The return type is calculated by the ``value_type`` type alias, which returns
the value type of the template argument. For our ``Raster`` type we can
implement all this as folows:


.. code-block:: cpp

   template<
       typename T>
   using value_type = TypeTraits<T>::value_type;

   template<
       typename Raster>
   value_type<Raster> const& get(
       Raster const& raster,
       size_t row,
       size_t col)
   {
       return raster[row][col];
   }

   template<
       typename Raster>
   value_type<Raster>& get(
       Raster& raster,
       size_t row,
       size_t col)
   {
       return raster[row][col];
   }


The last missing piece is the implementation of the TypeTraits class
template. Type traits provide properties of a type. In our case, we need
the value type of the ``Raster`` class.

.. code-block:: cpp

   template<
       typename T>
   struct TypeTraits
   {
   };

   template<>
   struct TypeTraits<
       Raster>
   {
       using value_type = double;
   };


Let's now rewrite our algorithm in terms of customization points.


.. code-block:: cpp

   template<
       typename Base,
       typename Exponent,
       typename Result>
   void pow(
       Base const& base,
       Exponent const& exponent,
       Result& result)
   {
       size_t const nr_rows{size(base, 0)}
       size_t const nr_cols{size(base, 1)}

       for(size_t r = 0; r < nr_rows; ++i) {
           for(size_t c = 0; c < nr_cols; ++j) {
               get(result, r, c) = std::pow(get(base, r, c), exponent);
           }
       }

       return result;
   }


With this in place, we can now call this algorithm with every raster-like
base argument for which we have implemented the three customization points.
In the next section we are going to get rid of the limitation that the base
argument must be a raster and the exponent a number.


Tag dispatching
---------------

