#pragma once


namespace fern {

//! Abstract base class for classes that model some form of uncertainty.
/*!
*/
class Uncertainty
{

public:

protected:

                   Uncertainty         ()=default;

    virtual        ~Uncertainty        ()=default;

                   Uncertainty         (Uncertainty&&)=default;

    Uncertainty&   operator=           (Uncertainty&&)=default;

                   Uncertainty         (Uncertainty const&)=default;

    Uncertainty&   operator=           (Uncertainty const&)=default;

private:

};

} // namespace fern
