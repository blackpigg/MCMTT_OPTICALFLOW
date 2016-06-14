#pragma once

#include "PSNWhere.h"
#include "Evaluator.h"
#include <queue>

struct stParamRunInput
{
	DATASET_TYPE datasetType;
	unsigned int seedValue;
};
struct stParamOutputInfo
{
	double score;
};
typedef std::pair<AlgorithmParams, std::vector<std::pair<stParamRunInput,stParamOutputInfo>>> paramSearchRecord;
struct cacheSearch
{
    cacheSearch(AlgorithmParams &s) : s_(s) { }
    bool operator()(const paramSearchRecord &p) const { return p.first == s_; }
    AlgorithmParams s_;
};

struct stParameterIndicator
{
	int position;
	int size;
};
typedef std::vector<stParameterIndicator> ParameterIndicatorSet;

class CParameterOptimizer
{
public:
	CParameterOptimizer(void);
	~CParameterOptimizer(void);

	AlgorithmParams Run(void);	
	AlgorithmParams LocalSearch(paramSearchRecord *startingParam, int minNumExecution);
	//AlgorithmParams Perturbation(void);
	bool RunAlgorithm(stParamRunInput runInput, AlgorithmParams params, stEvaluationResult &evaluationResult);
	

private:
	double GetPerformance(stEvaluationResult &result);
	double GetEstimatedPerformance(paramSearchRecord *records, int numEvaluations = 0);
	double GetPossibleBestPerformance(paramSearchRecord *records, int numTotalEvaluations);
	ParameterIndicatorSet GetParameterIndicator(AlgorithmParams *parameter);
	AlgorithmParams GetParameterFromIndicator(ParameterIndicatorSet &indicatorSet);

private:
	AlgorithmParams bestParams_;	
	DATASET_TYPE    nTargetDataSet_;
	std::list<paramSearchRecord> vecSearchCache_;
	ParameterIndicatorSet        vecBestParamIndicator_;
	cv::Rect_<double> rectCropZone_;
	double fCropZoneMargin_;
};


