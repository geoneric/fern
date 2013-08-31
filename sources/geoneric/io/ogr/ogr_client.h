#pragma once


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OGRClient
{

    friend class OGRClientTest;

public:

                   OGRClient           (OGRClient const&)=delete;

    OGRClient&     operator=           (OGRClient const&)=delete;

    virtual        ~OGRClient          ();

protected:

                   OGRClient           ();

private:

};

} // namespace geoneric
