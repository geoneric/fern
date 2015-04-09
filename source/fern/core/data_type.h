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

enum DataType {

    //! A single value, constant in time and space.
    DT_CONSTANT,

    //! A 2D field of spatially varying values, constant in time.
    DT_STATIC_FIELD,

    // DT_POINT,

    // DT_LINE,

    // DT_POLYGON,

    DT_LAST_DATA_TYPE = DT_STATIC_FIELD

};

} // namespace fern
