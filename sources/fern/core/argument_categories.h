#pragma once


namespace fern {

// Argument categories. Used in tag dispatching.
struct constant_tag {};
using array_0d_tag = constant_tag;  // Alias.
struct collection_tag {};
struct array_1d_tag: collection_tag {};
struct array_2d_tag: collection_tag {};
struct array_3d_tag: collection_tag {};
struct raster_1d_tag: array_1d_tag {};
struct raster_2d_tag: array_2d_tag {};
struct raster_3d_tag: array_3d_tag {};

} // namespace fern
