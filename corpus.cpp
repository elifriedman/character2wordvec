/*
 * Corpus.cpp
 *
 * Created:  11/20/2016
 * Author:  Eli Friedman (friedm3@cooper.edu)
 *
 */

#include <iostream>
#include "corpus.h"

Corpus::Corpus(const std::vector<std::string>& doc_names, const std::string& punc)
{
    // put punctuation into dict
    for(unsigned i=0; i < punc.size(); ++i) {
        std::string chr;
        chr += punc[i];
        punc_d.convert(chr);
    }
    punc_d.freeze();
    std::vector<std::string>::const_iterator it;
    for(it = doc_names.begin(); it != doc_names.end(); ++it) {
        std::ifstream doc(*it);
        words doc_contents = read_doc(doc);
        docs.push_back(doc_contents);
    }
}

Corpus::words Corpus::read_doc(std::ifstream& doc)
{
    words doc_contents;
    std::string word;
    while (doc >> word) {
        std::string cur_word;
        for(unsigned i = 0; i < word.size(); ++i) {
            std::string chr(1, word[i]);
            if (punc_d.contains(chr)) {
                if (cur_word.size() > 0) {
                    doc_contents.push_back(cur_word);
                    cur_word = "";
                }
                doc_contents.push_back(chr);
            }
            else {
                cur_word += chr;
            }
        }
        if (cur_word.size() > 0) {
            doc_contents.push_back(cur_word);
        }
    }

    return doc_contents;
}
