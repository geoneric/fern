#pragma once
#include <memory>
#include "geoneric/core/string.h"
#include "geoneric/feature/core/feature.h"
#include "geoneric/io/core/open_mode.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Dataset
{

public:

    //! Return the number of features in the dataset.
    /*!
        Nested features are not included in the count.
    */
    virtual size_t nr_features         () const=0;

    virtual bool   contains_feature    (String const& name) const=0;

    virtual bool   contains_attribute  (String const& name) const=0;

    virtual std::shared_ptr<Feature> read_feature(
                                        String const& name) const=0;

    virtual std::shared_ptr<Attribute> read_attribute(
                                        String const& name) const=0;

    virtual void   write_attribute     (Attribute const& attribute,
                                        String const& name) const=0;

protected:

                   Dataset             (String const& name,
                                        OpenMode open_mode);

                   Dataset             (Dataset const&)=delete;

    Dataset&       operator=           (Dataset const&)=delete;

                   Dataset             (Dataset&&)=delete;

    Dataset&       operator=           (Dataset&&)=delete;

    virtual        ~Dataset            ()=default;

    String const&  name                () const;

protected:

    OpenMode       open_mode           () const;

private:

    String         _name;

    OpenMode       _open_mode;

};

} // namespace geoneric
