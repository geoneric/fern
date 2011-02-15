********
Concepts
********
Concepts will be added to C++0x. Until then, concept requirements will be documented. Don't use Boost's Concept Check library because it will be replaced by C++0x's support for handling concepts.

Scalar
======

InputScalar concept
-------------------


Raster
======
Do we need different concepts of a raster, based on different requirements that algorithms have?

For example, an algorithm adding two rasters just needs a simple and fast way to visit all cells in the raster. This is comparable with the STL's input iterator concept, except that it should work in two dimensions. On the other hand, an algorithm summing the values in some neighborhood around each cell needs a way to visit all cells in the raster as well as a way to visit a group of cells in the neighborhood of each individual cell. So, the answer is yes.

Maybe we can base the Raster concept on the STL's iterator concept, with respect to terminology.

Scalars can be treated as rasters by indirection.

TrivialRaster concept
---------------------
A TrivialRaster is an object that can be asked for its dimensions.

.. code-block:: c++

   concept TrivialRaster<typename Raster>
   {
     typename value_type;
     typename difference_type;
     where SignedIntegral<difference_type>;

     difference_type nrRows            () const;
     difference_type nrCols            () const;
     difference_type nrCells           () const;
   };

InputRaster concept
-------------------
An InputRaster is an object that provides access to its cell values using a single cell index. The interface is optimized for accessing cells that are organized in a one-dimensional array, since this is the most likely data structure used for rasters. Alternative implementations may need to convert from the single index to specific row and col indices. The converse operation, converting from row and col indices to an array index is required for the RandomAccessRaster concept.

This concept is typically a requirement for algorithms implementing local operations.

.. code-block:: c++

   concept InputRaster<typename R>
     : TrivialRaster<Raster>
   {
     value_type    operator[]          (difference_type) const;
   };

RandomAccessRaster concept
--------------------------
A RandomAccessRaster is an object that provides access to its cells value using a row and column index.

This concept is typically a requirement for algorithms implementing neighborhood, zonal and global operations.

.. code-block:: c++

   concept InputRaster<typename R>
     : TrivialRaster<Raster>
   {
     value_type    cell                (difference_type row,
                                        difference_type col) const;
   };



