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
#include <map>
#include <tuple>


class Corpus
{
    public:
        Corpus(const std::vector<std::string>& doc_names, const std::string& punc);
        typedef std::vector<std::string> words;


        typedef std::string word;
        typedef std::string context;
        typedef bool is_real;
        typedef double prob_context;
        typedef std::tuple<word, context, is_real, prob_context> datapoint;
        std::vector<datapoint> makeDatasetNCE(int k);

    private:
        dynet::Dict punc_d; // dictionary of punctuation (used for fast lookup of punctuation)
        dynet::Dict  word_d; // maps words to word frequencies
        std::vector<double> wordfreq_d;
        std::vector<words> docs;

        words read_doc(std::ifstream& doc);
};


#endif // CORPUS_H
