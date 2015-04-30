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
#include "fern/core/expression_type.h"
#include "fern/core/path.h"
#include "fern/language/feature/core/feature.h"
#include "fern/language/io/core/open_mode.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Dataset
{

public:

    std::string const& name            () const;

    OpenMode       open_mode           () const;

    //! Return the number of features in the dataset.
    /*!
        Nested features are not included in the count.
    */
    virtual size_t nr_features         () const=0;

    virtual std::vector<std::string>
                   feature_names       () const=0;

    virtual bool   contains_feature    (Path const& path) const=0;

    virtual bool   contains_attribute  (Path const& path) const=0;

    virtual ExpressionType
                   expression_type     (Path const& path) const=0;

    //! Open a feature and return the result.
    /*!
      \param     path Path to the feature to be opened.
      \exception .
      \warning   .
      \sa        open_attribute(Path const&)

      Opening a feature differs from reading it in that no data is read. Only
      properties are read. It is important that opening an feature is
      efficient.
    */
    virtual std::shared_ptr<Feature>
                   open_feature        (Path const& path) const=0;

    //! Open an attribute and return the result.
    /*!
      \param     path Path to the attribute to be opened.
      \exception .
      \warning   .
      \sa        open_feature(Path const&)

      Opening an attribute differs from reading it in that no data is read.
      Only properties are read. It is important that opening an attribute is
      efficient.
    */
    virtual std::shared_ptr<Attribute>
                   open_attribute      (Path const& path) const=0;

    virtual std::shared_ptr<Feature>
                   read_feature        (Path const& path) const=0;

    virtual std::shared_ptr<Attribute>
                   read_attribute      (Path const& path) const=0;

    virtual void   write_attribute     (Attribute const& attribute,
                                        Path const& path)=0;

protected:

                   Dataset             (std::string const& name,
                                        OpenMode open_mode);

                   Dataset             (Dataset const&)=delete;

    Dataset&       operator=           (Dataset const&)=delete;

                   Dataset             (Dataset&&)=delete;

    Dataset&       operator=           (Dataset&&)=delete;

    virtual        ~Dataset            ()=default;

private:

    std::string    _name;

    OpenMode       _open_mode;

};

} // namespace fern
