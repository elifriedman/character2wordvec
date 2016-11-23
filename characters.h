/*
 * characters.h
 *
 * Created:  11/20/2016
 * Author:  Eli Friedman (friedm3@cooper.edu)
 *
 */
#ifndef CHARACTERS_H
#define CHARACTERS_H

#include "dynet/dict.h"
#include <iostream>
#include <fstream>

class Characters
{
    public:
        Characters(const std::string& charfilename);
        const std::string SOW = "<w>";
        const std::string EOW = "</w>";

        inline
        std::vector<int> word2idxvec(const std::string& word) 
        {
            std::vector<int> idxvec;
            idxvec.push_back(convert(SOW));
            std::string::const_iterator it;
            for(it = word.begin(); it != word.end(); ++it) {
                idxvec.push_back(convert(*it));
            }
            idxvec.push_back(convert(EOW));
            return idxvec;
        }

        inline
        int convert(const std::string& character)
        {
            return chars_d.convert(character);
        }

        inline
        int convert(char character)
        {
            return chars_d.convert(std::string(1, character));
        }

        inline
        std::string idx2char(int i)
        {
            return chars_d.convert(i);
        }

        inline
        unsigned size()
        {
            return chars_d.size();
        }
    private:
        dynet::Dict chars_d;
};

#endif // CHARACTERS_H
