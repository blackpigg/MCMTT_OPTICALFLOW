#pragma once

#include "PSNWhere.h"
#include "Evaluator.h"

class CParameterOptimizer
{
public:
	CParameterOptimizer(void);
	~CParameterOptimizer(void);

	AlgorithmParams Run(void);
	stEvaluationResult RunAlgorithm(DATASET_TYPE datasetType, AlgorithmParams params);

private:
	AlgorithmParams currentParams_;
	DATASET_TYPE nTargetDataSet_;	

	std::vector<AlgorithmParams> paramCache_;
};

