#pragma once
#include <memory>
#include "geoneric/core/string.h"
#include "geoneric/feature/core/feature.h"


namespace geoneric {

enum class OpenMode { READ, WRITE, UPDATE };


//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Dataset
{

public:

    virtual size_t nr_features         () const=0;

    virtual bool   contains_feature    (String const& name) const=0;

    virtual std::shared_ptr<Feature> read(String const& name) const=0;

protected:

                   Dataset             (String const& name,
                                        OpenMode open_mode);

                   Dataset             (Dataset const&)=delete;

    Dataset&       operator=           (Dataset const&)=delete;

                   Dataset             (Dataset&&)=delete;

    Dataset&       operator=           (Dataset&&)=delete;

    virtual        ~Dataset            ()=default;

    String const&  name                () const;

private:

    String         _name;

    OpenMode       _open_mode;

};

} // namespace geoneric
