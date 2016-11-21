/*
 * characters.cpp
 *
 * Created:  11/20/2016
 * Author:  Eli Friedman (friedm3@cooper.edu)
 *
 */

#include "characters.h"

Characters::Characters(const std::string& charfilename)
{
    std::ifstream ifs(charfilename);
    char c;
    while (ifs.get(c)) {
        if (c == '\n') continue;
        std::string chr = "";
        chr += c;
        convert(chr); // add to dictionary
    }
    convert(SOW);
    convert(EOW);
    chars_d.freeze();
}
