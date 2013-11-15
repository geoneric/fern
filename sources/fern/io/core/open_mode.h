#pragma once


namespace fern {

enum class OpenMode {
    //! Open dataset for reading.
    READ,

    //! Open dataset for reading and writing. Dataset will be truncated.
    OVERWRITE,

    //! Open dataset for reading and writing. Dataset will not be truncated.
    UPDATE
};

} // namespace fern
