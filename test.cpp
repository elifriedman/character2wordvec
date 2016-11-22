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

int main(int argc, char** argv) {

    Characters chars("characters.txt");

    std::vector<std::string> fnames;
    fnames.push_back("data/doc1");
    fnames.push_back("data/doc2");
    Corpus docs(fnames, ".,()`\"");

    dynet::initialize(argc, argv);
    std::vector<Corpus::datapoint> dataset = docs.makeDatasetNCE(0);
    std::cout << dataset.size() << "\n";
//
//    // parameters
    ComputationGraph cg;
    Model model;
//
//    BiRNNModel<LSTMBuilder> bilstm(model,
//                                   1, // # layers
//                                   2, // input dim
//                                   5, // hidden dim
//                                   3, // output dim
//                                   chars.size()); // # chars
//
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

