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
namespace language {

enum class OpenMode {
    //! Open dataset for reading.
    READ,

    //! Open dataset for reading and writing. Dataset will be truncated.
    OVERWRITE,

    //! Open dataset for reading and writing. Dataset will not be truncated.
    UPDATE
};

} // namespace language
} // namespace fern
