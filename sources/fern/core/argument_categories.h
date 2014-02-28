#pragma once


namespace fern {

// Argument categories. Used in tag dispatching.
struct constant_tag {};
struct collection_tag {};
struct array_1d_tag: collection_tag {};
struct array_2d_tag: collection_tag {};
struct array_3d_tag: collection_tag {};

} // namespace fern
