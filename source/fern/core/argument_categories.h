// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
