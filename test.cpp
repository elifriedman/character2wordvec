#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/dict.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>

#include "characters.h"
#include "corpus.h"
#include "model.h"
#include <random>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

template <class Builder>
Expression calculateNCELoss(std::vector<Corpus::datapoint>::const_iterator& begin,
                            std::vector<Corpus::datapoint>::const_iterator& end,
                            int k,
                            Characters& chars,
                            BiRNNModel<Builder>& model,
                            ComputationGraph& cg)
{
    std::vector<Corpus::datapoint>::const_iterator it;
//    Expression loss = 0;
    for(it = begin; it != end; ++it) {
        const std::string& word = std::get<Corpus::WORD_IDX>(*it);
        std::vector<int> wordVec = chars.word2idxvec(word);

        const std::string& context = std::get<Corpus::CONTEXT_IDX>(*it);
        std::vector<int> contextVec = chars.word2idxvec(context);

        bool is_real = std::get<Corpus::IS_REAL_IDX>(*it);
        double probability = std::get<Corpus::PROB_CONTEXT_IDX>(*it);

        Expression output = model.getNCEModelOutput(wordVec, contextVec, cg);

        if (is_real) {
            // TODO loss += output / (output + k*probability)
        }
        else {
            // TODO loss += k*probability / (output + k*probability)
        }
//        std::cout << std::get<Corpus::WORD_IDX>(*it) << " "
//                  << std::get<Corpus::CONTEXT_IDX>(*it) << " "
//                  << std::get<Corpus::IS_REAL_IDX>(*it) << " "
//                  << std::get<Corpus::PROB_CONTEXT_IDX>(*it) << "\n";
    }

    return loss;
}

int main(int argc, char** argv) {

    Characters chars("characters.txt");

    std::vector<std::string> fnames;
    fnames.push_back("data/doc1");
    fnames.push_back("data/doc2");
    Corpus docs(fnames, ".,()`\"");

    dynet::initialize(argc, argv);
    std::vector<Corpus::datapoint> dataset = docs.makeDatasetNCE(3);
    
    std::vector<Corpus::datapoint>::const_iterator it;
    for(it = dataset.begin(); it != dataset.end(); ++it) {
//        std::cout << std::get<Corpus::WORD_IDX>(*it) << " "
//                  << std::get<Corpus::CONTEXT_IDX>(*it) << " "
//                  << std::get<Corpus::IS_REAL_IDX>(*it) << " "
//                  << std::get<Corpus::PROB_CONTEXT_IDX>(*it) << "\n";
    }
//
//    // parameters
    ComputationGraph cg;
    Model model;

    bool use_momentum = true;
    Trainer* sgd = nullptr;
    if (use_momentum)
      sgd = new MomentumSGDTrainer(&model);
    else
      sgd = new SimpleSGDTrainer(&model);

    BiRNNModel<LSTMBuilder> bilstm(model,
                                   1, // # layers
                                   2, // input dim
                                   5, // hidden dim
                                   3, // output dim
                                   chars.size()); // # chars

//    unsigned report_every_i = 50;
//    unsigned dev_every_i_reports = 25;
//    unsigned si = training.size();
//    vector<unsigned> order(training.size());
//    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
//    bool first = true;
//    int report = 0;
//    unsigned lines = 0;
//    while(1) {
//      Timer iteration("completed in");
//      double loss = 0;
//      for (unsigned i = 0; i < report_every_i; ++i) {
//        if (si == dataset.size()) {
//            si = 0;
//            if (first) { first = false; } else { sgd->update_epoch(); }
//            shuffle(order.begin(), order.end(), *rndeng);
//        }
//  
//        // build graph for this instance
//        ComputationGraph cg;
//        auto& sent = training[order[si]];
//        ++si;
//      }
//    }

//    std::string text = "hello!";
//    std::vector<int> input;
//    for(unsigned i = 0; i < text.size(); ++i) {
//        int idx = chars.convert(text[i]);
//        std::cout << text[i] << " - " << idx << "\n";
//        input.push_back(idx);
//    }
//
//    Expression output = bilstm.build(input, cg);
//    std::cout << cg.forward(output) << "\n";
//  SimpleSGDTrainer sgd(&model);
//  //MomentumSGDTrainer sgd(&m);
//
//  ComputationGraph cg;
//  LookupParameter p = model.add_lookup_parameters(2, {5});
//
//  unsigned idx = 0;
//  Expression x = lookup(cg, p, idx);
//
//  cg.print_graphviz();
//  auto vec = as_vector(cg.forward(x));
//  for(unsigned i=0; i < vec.size(); ++i) {
//      cout << vec[i] << ",";
//  }
//  cout << "\n";
}

