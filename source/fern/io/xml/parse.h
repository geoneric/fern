// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include <fstream>
#include <fern/core/io_error.h>
#include <fern/core/parse_error.h>
#include <fern/io/core/file.h>


namespace fern {
namespace io {
namespace xml {

/*!
    @ingroup    fern_io_xml_group
    @brief      Call @a parse to parse the XML document pointed to by
                @a pathname and return the contents in a tree-like
                datastructure.
    @tparam     Parser Function that accepts an std::istream for reading and
                returns a datastructure representing the XML document.
    @exception  fern::IOError In case file @a pathname does not exist.
    @exception  fern::IOError In case file @a pathname cannot be read.
    @exception  fern::ParseError In case the document in file @a pathname
                is not valid.
*/
template<
    typename Parser>
inline auto parse(
    Parser const& parse,
    std::string const& pathname) ->
        typename std::result_of<Parser(std::istream&)>::type
{
    if(!file_exists(pathname)) {
        throw IOError(pathname,
            Exception::messages()[MessageId::DOES_NOT_EXIST]);
    }

    std::ifstream stream(pathname, std::ios_base::in);

    if(!stream) {
        throw IOError(pathname,
            Exception::messages()[MessageId::CANNOT_BE_READ]);
    }


    using Document = typename std::result_of<Parser(std::istream&)>::type;

    Document document;

    try {
        // xml_schema::flags::dont_validate
        document = parse(stream);
    }
    catch(xml_schema::parsing const& exception) {
        assert(!exception.diagnostics().empty());

        throw ParseError(pathname,
            exception.diagnostics()[0].line(),
            exception.diagnostics()[0].column(),
            exception.diagnostics()[0].message());
    }


    assert(document);
    return document;
}

} // namespace xml
} // namespace io
} // namespace fern
