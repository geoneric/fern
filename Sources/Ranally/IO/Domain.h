#pragma once


namespace ranally {

//! A Domain positions a Feature's Attribute in the spatio-temporal coordinate space.
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Domain
{

    friend class DomainTest;

public:

    enum Type {
        PointDomain,
        PolygonDomain
    };

                   Domain              (Domain const&)=delete;

    Domain&        operator=           (Domain const&)=delete;

    virtual        ~Domain             ();

    Type           type                () const;

    virtual bool   isSpatial           () const=0;

    virtual bool   isTemporal          () const=0;

protected:

                   Domain              (Type type);

private:

  Type             _type;

};

} // namespace ranally
