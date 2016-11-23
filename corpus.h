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

        typedef std::string word;         static const unsigned WORD_IDX = 0;
        typedef std::string context;      static const unsigned CONTEXT_IDX = 1;
        typedef bool        is_real;       static const unsigned IS_REAL_IDX = 2;
        typedef double      prob_context;  static const unsigned PROB_CONTEXT_IDX = 3;
        typedef std::tuple<word, context, is_real, prob_context> datapoint;

        std::vector<datapoint> makeDatasetNCE(int k);

        typedef std::vector<word> words;
    private:
        dynet::Dict punc_d; // dictionary of punctuation (used for fast lookup of punctuation)
        dynet::Dict  word_d; // maps words to word frequencies
        std::vector<double> wordfreq_d;
        std::vector<words> docs;

        words read_doc(std::ifstream& doc);
        std::vector<int> word2intvector(const std::string& word);
};


#endif // CORPUS_H
