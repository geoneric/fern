Requirements
============
In this section we will develop a first version of a generic algorithm
handling 2D spatial field data. It is good practice to start with a
non-generic version, and get a clear idea about the shortcomings, before
generalizing the code.

For our purposes here we are not interested in the calculations
themselves. In fact we are interested in everything but the calculations
themselves. The algorithm we will develop will raise a 2D spatial field
to the power of some exponent. Here is a first try:

.. literalinclude:: source/pow_1/pow.cc
   :language: cpp

We use the ``Raster`` class template from `Geoneric`_'s Fern library, but
this could be any class with which we can model a raster. The details
of this class' implementation don't matter here.

Some observations and drawbacks of this implementation:

- The type of the individual elements (further on we call this the `value
  type`) in the base raster are fixed to ``double``. This is limiting the
  caller. She might prefer to pass in ``float`` or ``long double``
  values. The same is true for the exponent.
- The type of the argument itself (the `data type`) is fixed to ``Raster``.
  This is also limiting the caller. She has to use our ``Raster`` type
  while she may have a type for modelling rasters that fits her
  needs better.
- The exponent is fixed to a ``double``, but we may want to be able to
  pass a raster instead of a single number. When we pass a exponent raster
  we expect each cell in the base raster to be raised by the corresponding
  cell in the exponent raster.
- The result is returned, which is OK if ``Raster`` supports move semantics.
  Otherwise it is definitely not OK. Furthermore, in C++ return types are
  not considered in overload resolution which limits our possibilities to
  make the return type configurable by the caller. This will be explained
  in more detail later on.
- Often, algorithms have a limited domain of valid input values. For
  example, a negative finite base and a non-integer finite exponent are
  considered out of domain by the ``std::pow`` algorithm. What should
  happen if an algorithm's argument(s) contains values that are not part of
  the algorithm's valid domain of argument values?
- Similarly, result value types have a limited range of values they can
  represent. For example, an ``int8_t`` cannot hold values larger than
  :math:`(2^8 / 2) - 1`. What should happen if an algorithm calculates
  values that are not part of the valid range of values?
- In the algorithm above, we perform a calculation for each cell. In reality,
  raster often have cells with undefined values. Depending on the field
  that is represented by the raster, this may be due to clouds, measurement
  errors, previous calculations, etc. We need a way to test whether or not
  a cell from a raster argument contains a valid value or not.
- Given the previous observations, we need a way to mark that a result
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
  from a fixed set of types. The user can craft her own types.
- It must be possible for the user to `explain` to the algorithm how to

      - detect invalid argument raster cell values
      - mark invalid result raster cell values
      - handle out of domain argument values
      - handle out of range result values

In the next sections, we are going to tackle each of these requirements
in turn.


.. _Geoneric: http://www.geoneric.eu
