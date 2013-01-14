#pragma once


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Domain
{

    friend class DomainTest;

public:

    virtual        ~Domain             ()=default;

protected:

                   Domain              ()=default;

                   Domain              (Domain const&)=delete;

    Domain&        operator=           (Domain const&)=delete;

                   Domain              (Domain&&)=delete;

    Domain&        operator=           (Domain&&)=delete;

private:

};

} // namespace ranally
