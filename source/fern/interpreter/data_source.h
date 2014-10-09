#pragma once
#include "fern/core/expression_type.h"


namespace fern {

class Argument;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        DataSync
*/
class DataSource
{

public:

    virtual ExpressionType const&
                   expression_type     () const=0;

    virtual std::shared_ptr<Argument>
                   read                () const=0;

protected:

                   DataSource          ();

                   DataSource          (DataSource const&)=delete;

    DataSource&    operator=           (DataSource const&)=delete;

                   DataSource          (DataSource&&)=delete;

    DataSource&    operator=           (DataSource&&)=delete;

    virtual        ~DataSource         ()=default;

#ifndef NDEBUG
    bool           data_has_been_read_already
                                       () const;

    void           set_data_has_been_read
                                       () const;
#endif

private:

#ifndef NDEBUG
    //! Whether the data is read already.
    mutable bool   _read;
#endif

};

} // namespace fern
