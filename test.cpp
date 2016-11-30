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
double calculateNCELoss(std::vector<Corpus::datapoint>::const_iterator begin,
                            std::vector<Corpus::datapoint>::const_iterator end,
                            int k,
                            Characters& chars,
                            BiRNNModel<Builder>& model,
                            ComputationGraph& cg,
                            int* accuracy,
                            double* output_scalar)
{
    std::vector<Expression> losses;

    std::vector<Corpus::datapoint>::const_iterator it;
    double size = end - begin;
    for(it = begin; it != end; ++it) {

        const std::string& word = std::get<Corpus::WORD_IDX>(*it);
        std::vector<int> wordVec = chars.word2idxvec(word);

        const std::string& context = std::get<Corpus::CONTEXT_IDX>(*it);
        std::vector<int> contextVec = chars.word2idxvec(context);

        bool is_real = std::get<Corpus::IS_REAL_IDX>(*it);
        Expression probability = input(cg, std::get<Corpus::PROB_CONTEXT_IDX>(*it));

        Expression output = model.getNCEModelOutput(wordVec, contextVec, cg);

        Expression loss;
        if (is_real) {
            loss = 1 - output; //cdiv(output,  (output + k*probability));
        }
        else {
            loss = output; //cdiv(k*probability, (output + k*probability));
        }
        losses.push_back(loss);

        *output_scalar = as_scalar(cg.incremental_forward(output));
        bool predict = *output_scalar >= 0.5;
//        std::cout << word << " : " << context << " . " << predict << " -- " << is_real << "\n";
        *accuracy += (predict == is_real);
    }

    Expression total_loss = sum(losses);
    double loss_scalar = as_scalar(cg.incremental_forward(total_loss));
    cg.backward(total_loss);
    return loss_scalar;
}

int main(int argc, char** argv) {

    Characters chars("characters.txt");

    std::vector<std::string> fnames;
    fnames.push_back("data/doc1");
    fnames.push_back("data/doc2");
    Corpus docs(fnames, ".,()`\"");

    dynet::initialize(argc, argv);
    std::cout << "Loading dataset...\n";
    std::vector<Corpus::datapoint> dataset = docs.makeDatasetNCE(3);
    std::cout << "Loaded dataset.\n";
    
    Model model;
    BiRNNModel<LSTMBuilder> bilstm(model,
                                   1, // # layers
                                   16, // input dim
                                   128, // hidden dim
                                   128, // output dim
                                   chars.size()); // # chars


    bool use_momentum = false;
    Trainer* sgd = nullptr;
    if (use_momentum)
      sgd = new MomentumSGDTrainer(&model);
    else
      sgd = new AdagradTrainer(&model);
//      sgd = new SimpleSGDTrainer(&model);

    int iteration_num = 0;
    unsigned batch_size = 100;
    bool first = true;
    std::cout << "size = " << dataset.size() << "\n";
    
    while(1) {
        double loss_scalar = 0;
        int accuracy = 0;
        double avg_output = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        shuffle(dataset.begin(), dataset.end(), *rndeng);
        std::cout << "Iteration #" << iteration_num << ":\n";
        std::vector<Corpus::datapoint>::const_iterator it = dataset.begin();
        unsigned i = 0;
        for (; i < dataset.size(); it += batch_size, i += batch_size) {
            // build graph for this instance
            std::cout << i << ": ";
            ComputationGraph cg;
            bilstm.new_graph(cg);
            int cur_accuracy = 0;
            double cur_output = 0;
            double cur_loss = calculateNCELoss(it, it + batch_size,
                                               1, // k
                                               chars, bilstm, cg,
                                               &cur_accuracy, &cur_output);
            accuracy += cur_accuracy;
            loss_scalar += cur_loss;
            std::cout  << "curr_loss: " << cur_loss
                       << ", accuracy: " << cur_accuracy
                       << ", avg_output: " << cur_output / batch_size;
                      
            std::cout << ", grad_l2_norm: " << model.gradient_l2_norm()
                      << "\n";
            sgd->update();
       }

        std::cout << "Loss = " << loss_scalar / dataset.size() 
                  << ", Accuracy = " << ((double)accuracy) / dataset.size()
                  << "\n";
        iteration_num++;
    }
}

