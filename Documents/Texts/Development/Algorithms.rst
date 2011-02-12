**********
Algorithms
**********

Requirements
============
* Algorithms must be fast as hell.
* Support for different implementations of the same algorithm. For example, to add two ranges, their may be two implementations: one simple one in C++, and one using OpenCL. This is called lifting in generic programming.
* Have a generic interface that supports many argument types, including user-provided ones, ranges, values. Data structures representing rasters can be implemented in various ways, including one- and two-dimensional arrays, vectors, etc.
* Configurable with respect to handling of missing values / no-data.
  * There must be no performance penalty, if the user doesn't want to handle missing values.
* Configurable with respect to handling of argument domain errors.
  * There must be no performance penalty, if the user doesn't want to handle domain issues.
* Configurable with respect to handling of result range errors.
  * Integers wrap, floating points have infinity.
  * There must be no performance penalty, if the user doesn't want to handle out of range issues.

Classification
==============
Unary, Binary, NAry.

.. todo::

   Describe algorithm classification.

Design
======

.. todo::

   For inspiration, check design page in Boost.Geometry library docs.

.. code-block:: c++

   template<class Range1, class Range2, class Range3>
   void binaryAlgorithm(
     Range1 const& argument1,
     Range2 const& argument2,
     Range3& result)









Implementation
==============
* OpenCL
* OpenMP
* Threading

