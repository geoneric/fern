// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
// #include "fern/core/data_name.h"
#include "fern/io/core/dataset.h"
#include "fern/io/core/open_mode.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Driver
{

public:

    //! Return whether the driver can open \a name in \a open_mode.
    /*!
      \param     name Name of dataset to open.
      \param     open_mode Open mode to use.
      \warning   In case open_mode is OVERWRITE, this function returns true
                 if a new dataset could potentially be created. It doesn't
                 mean that \a name exists.

      This function aims to be as lightweight as possible, performing only the
      minimum amount of tests, while still being able to tell whether the
      dataset can be opened successfully or not.
    */
    virtual bool   can_open            (String const& name,
                                        OpenMode open_mode)=0;

    // virtual ExpressionType
    //                expression_type     (DataName const& data_name) const=0;

    //! Open \a name in \a open_mode and return the resulting dataset.
    /*!
      \param     name Name of dataset to open.
      \param     open_mode Open mode to use.
      \warning   In case open_mode is OVERWRITE, the dataset returned may not
                 point to an existing dataset yet.
    */
    virtual std::shared_ptr<Dataset> open(
                                        String const& name,
                                        OpenMode open_mode)=0;

protected:

                   Driver             (String const& name);

                   Driver             (Driver const&)=delete;

    Driver&       operator=           (Driver const&)=delete;

                   Driver             (Driver&&)=delete;

    Driver&       operator=           (Driver&&)=delete;

    virtual        ~Driver            ()=default;

public:

    String const&  name               () const;

private:

    String         _name;

};

} // namespace fern
