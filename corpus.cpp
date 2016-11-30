/*
 * Corpus.cpp
 *
 * Created:  11/20/2016
 * Author:  Eli Friedman (friedm3@cooper.edu)
 *
 */

#include <iostream>
#include <random>
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
    std::map<int, double> freqs;
    double total_sum = 0;
    for(it = doc_names.begin(); it != doc_names.end(); ++it) {
        std::ifstream doc(*it);
        words doc_contents = read_doc(doc);
        docs.push_back(doc_contents);

        words::const_iterator wordItr;
        for (wordItr = doc_contents.begin(); wordItr != doc_contents.end(); ++wordItr) {
            freqs[word_d.convert(*wordItr)] += 1;
            total_sum += 1;
        }
    }

    std::map<int, double>::const_iterator mapItr;
    for (mapItr = freqs.begin(); mapItr != freqs.end(); ++mapItr) {
        double frequency = mapItr->second;
        wordfreq_d.push_back(frequency / total_sum);
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

std::vector<Corpus::datapoint> Corpus::makeDatasetNCE(int k)
{
    std::default_random_engine generator; // TODO choose seed
    std::discrete_distribution<int> choose_word(wordfreq_d.begin(), wordfreq_d.end());

    std::vector<datapoint> dataset;

    std::vector<words>::const_iterator it;
    for (it = docs.begin(); it != docs.end(); ++it) {

        words::const_iterator wordItr;
        for (wordItr = it->begin(); wordItr != it->end()-1; ++wordItr) {

            std::string word = *wordItr;    std::string context = *(wordItr+1);

            double probability = wordfreq_d[word_d.convert(word)]; // get q(word)

            datapoint d = std::make_tuple(word, context, true, probability);
            dataset.push_back(d);


            for (int i = 0; i < k; ++i) { // make "fake" contexts
                unsigned fake_context_idx = choose_word(generator);
                std::string fake_context = word_d.convert(fake_context_idx);

                datapoint fake_d = std::make_tuple(word, fake_context, false, probability);
                dataset.push_back(fake_d);
            }
        }
    }

    return dataset;
}
