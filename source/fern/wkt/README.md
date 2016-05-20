Spirit X3
- No grammars anymore, just compose rules
- Only use auto for non-recursive rules




[Geographic information - Well-known text representation of coordinate reference systems](http://docs.opengeospatial.org/is/12-063r5/12-063r5.html)

- The WKT string is a representation of the definition of a CRS or coordinate operation. A string describes one CRS or coordinate operation object. Each object is represented by a token comprised of a keyword followed by a set of attributes of the object, the set enclosed by delimiters. Some objects are composed of other objects so the result may be a nested structure. Nesting may continue to any depth.
- Keywords are case-insensitive.
- The delimiters are normally <left bracket> and <right bracket>. Implementations are free to substitute parentheses for brackets.
- Attributes may be from an enumeration, be numbers or be text. Text is enclosed in double quotes. Two forms of text are defined, one restricted to the Latin1 character set and the other permitting any Unicode character set. Attributes are separated by a comma.
- A WKT string contains no white space outside of double quotes. However padding with white space to improve human readability is permitted. Any padding is stripped out or ignored by parsers
- All WKT strings are realized as a sequence of characters, or a character string. The only restriction is that the same encoding shall be used throughout the entire WKT definition.
    - A WKT string shall use one encoding throughout the entire string.
    - The characters used in a WKT string shall be wholly contained within the domain of a specific character set. This character set shall exist as a subset of the repertoire of the Universal Character Set specified by ISO 10646:2012.

