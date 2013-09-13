#pragma once


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Domain
{

public:

                   Domain              ()=default;

                   Domain              (Domain const&)=delete;

    Domain&        operator=           (Domain const&)=delete;

                   Domain              (Domain&&)=delete;

    Domain&        operator=           (Domain&&)=delete;

                   ~Domain             ()=default;

private:

};

} // namespace geoneric
