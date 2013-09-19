#pragma once


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Values
{

public:

protected:

                   Values              ()=default;

                   Values              (Values const&)=delete;

    Values&        operator=           (Values const&)=delete;

                   Values              (Values&&)=delete;

    Values&        operator=           (Values&&)=delete;

    virtual        ~Values             ()=default;

private:

};

} // namespace geoneric
