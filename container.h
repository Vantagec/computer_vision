#ifndef CONTAINER_H
#define CONTAINER_H

#include <cmath>

namespace mnist {

template <typename Container>
void binarize_each(Container& values, double threshold = 30.0) {
    for (auto& vec : values) {
        for (auto& v : vec) {
            v = v > threshold ? 1.0 : 0.0;
        }
    }
}

template <typename Container>
double mean(const Container& container) {
    double mean = 0.0;
    for (auto& value : container) {
        mean += value;
    }
    return mean / container.size();
}


template <typename Container>
double stddev(const Container& container, double mean) {
    double std = 0.0;
    for (auto& value : container) {
        std += (value - mean) * (value - mean);
    }
    return std::sqrt(std / container.size());
}


template <typename Container>
void normalize_each(Container& values) {
    for (auto& vec : values) {
        //zero-mean
        auto m = mnist::mean(vec);
        for (auto& v : vec) {
            v -= m;
        }
        //unit variance
        auto s = mnist::stddev(vec, 0.0);
        for (auto& v : vec) {
            v /= s;
        }
    }
}


template <typename Dataset>
void binarize_dataset(Dataset& dataset) {
    mnist::binarize_each(dataset.training_images);
    mnist::binarize_each(dataset.test_images);
}


template <typename Dataset>
void normalize_dataset(Dataset& dataset) {
    mnist::normalize_each(dataset.training_images);
    mnist::normalize_each(dataset.test_images);
}



}
#endif // CONTAINER_H
