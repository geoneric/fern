Abstraction 2: no-data
======================
The algorithm we developed until now assumes all argument raster cells
contain valid data. In reality, this is often not the case. Ideally, we
want to be able to tell the algorithm whether or not our argument rasters
contain invalid cells or not, and if so, how to test each individual cell
for validity.

Similarly, we want to be able to mark the corresponding cells in the
result raster as invalid, for example in case one of the argument rasters
contains invalid cells.

`Policy classes`_ are a technique that can help us with these
requirements. They allow us to parameterize behavior of an algorithm,
which enables the caller to tune the algorithm to her needs. An example
of a policy from the C++ Standard Library is `Allocator`_, which allows
the caller to tune the allocation and deallocation of memory in code
accepting an allocator policy (most containers from the Standard Library do, see for example the second template argument of the `std::vector`_ class template).

Our algorithm needs to be parameterized with policy classes
that encapsulate the behavior of how to deal with no-data values in the
argument and result rasters. We must rewrite our algorithm to use the
no-data testing policy class for testing individual raster elements. The
caller determines if and how these tests are performed.  Depending on
the raster type used, no-data may be marked by a special marker value,
like the maximum or minimum representable value by the value type, or
maybe by a layered boolean raster or a bit mask. It all depends on the
type passed in the algorithm.



..
   If the argument rasters do not handle invalid cells, we do not want our algorithm to spend CPU cycles testing them.

   In case the raster doesn't have support for representing invalid elements, a dummy policy class can be implemented that does not affect the algorithm's efficiency.


Enough theory, let us think about the interface of the no-data checking
policy class. Its sole 















.. _Policy classes: http://www.boost.org/community/generic_programming.html#policy
.. _Allocator: http://en.cppreference.com/w/cpp/concept/Allocator
.. _std::vector: http://en.cppreference.com/w/cpp/container/vector
