#ifndef CLASSIFIERFASHION_H
#define CLASSIFIERFASHION_H

#pragma once

#include <vector>

namespace ml_fahion
{

class IclassifierFashion
{
public:
    using features_t = std::vector<float>;

    virtual ~IclassifierFashion() {}

    virtual float predict_proba(const features_t&) const = 0;
};


class LogregClassifier: public IclassifierFashion {
public:
    using coef_t = features_t;

    LogregClassifier(const coef_t& coef);

    float predict_proba(const features_t& feat) const override;

protected:
    std::vector<float> coef_;
};


}

#endif // CLASSIFIERFASHION_H
