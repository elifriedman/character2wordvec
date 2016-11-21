/*
 * corpus.h
 *
 * Created:  11/20/2016
 * Author:  Eli Friedman (friedm3@cooper.edu)
 *
 */
#ifndef CORPUS_H
#define CORPUS_H

#include "dynet/dict.h"
#include <fstream>
#include <vector>


class Corpus
{
    public:
        Corpus(const std::vector<std::string>& doc_names, const std::string& punc);

        typedef std::vector<std::string> words;

    private:
        dynet::Dict punc_d;
        std::vector<words> docs;

        words read_doc(std::ifstream& doc);
};


#endif // CORPUS_H
