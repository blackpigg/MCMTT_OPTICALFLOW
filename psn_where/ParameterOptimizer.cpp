#include "ParameterOptimizer.h"

#define NUM_PARAMS (16)
#define MIN_NUM_EVALUATION (1)

// 2D tracking parameter domains
const double nBackTrackingInterval[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const double fFeatureWindowSizeRatio[] = {0.1, 0.2, 0.3, 0.5, 0.7, 1.0};

// 3D association parameter domains
const double nProcWindowSize[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
const double nKBestSize[] = {1, 5, 10, 15, 20, 25, 30};
const double nMaxTrackInOptimization[] = {100, 500, 1000, 1500, 2000, 2500, 3000};
const double nMaxTrackInConfirmedTrackTree[] = {10, 50, 100, 200, 300};
const double nMaxTrackInUnconfirmedTrackTree[] = {2, 3, 4, 5, 10};
const double nNumFrameForConfirmation[] = {1, 2, 3, 4, 5};

const double fMaxMovingSpeed[] = {500, 900, 1000, 2000};

const double fProbEnterMax[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fProbEnterDecayCoef[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fCostEnterMax[] = {100, 200, 500, 1000, 1500};

const double fProbExitMax[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fProbExitDecayCoef_dist[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fProbExitDecayCoef_length[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fCostExitMax[] = {100, 200, 500, 1000, 1500};

const int fixedParameter[] = {2, 3};

const double* paramDomain[] = {
	nBackTrackingInterval,
	fFeatureWindowSizeRatio,
	nProcWindowSize,
	nKBestSize,
	nMaxTrackInOptimization,
	nMaxTrackInConfirmedTrackTree,
	nMaxTrackInUnconfirmedTrackTree,
	nNumFrameForConfirmation,
	fMaxMovingSpeed,
	fProbEnterMax,
	fProbEnterDecayCoef,
	fCostEnterMax,
	fProbExitMax,
	fProbExitDecayCoef_dist,
	fProbExitDecayCoef_length,
	fCostExitMax
};

CParameterOptimizer::CParameterOptimizer(void)
{
	vecBestParamIndicator_.resize(NUM_PARAMS);

	vecBestParamIndicator_[0].position = 3;
	vecBestParamIndicator_[1].position = 1;
	vecBestParamIndicator_[2].position = 8;
	vecBestParamIndicator_[3].position = 2;
	vecBestParamIndicator_[4].position = 4;
	vecBestParamIndicator_[5].position = 2;
	vecBestParamIndicator_[6].position = 2;
	vecBestParamIndicator_[7].position = 2;
	vecBestParamIndicator_[8].position = 1;

	vecBestParamIndicator_[9].position = 1;
	vecBestParamIndicator_[10].position = 3;
	vecBestParamIndicator_[11].position = 4;

	vecBestParamIndicator_[12].position = 2;
	vecBestParamIndicator_[13].position = 3;
	vecBestParamIndicator_[14].position = 1;
	vecBestParamIndicator_[15].position = 4;
	
	vecBestParamIndicator_[0].size = psn::size(nBackTrackingInterval);
	vecBestParamIndicator_[1].size = psn::size(fFeatureWindowSizeRatio);
	vecBestParamIndicator_[2].size = psn::size(nProcWindowSize);
	vecBestParamIndicator_[3].size = psn::size(nKBestSize);
	vecBestParamIndicator_[4].size = psn::size(nMaxTrackInOptimization);
	vecBestParamIndicator_[5].size = psn::size(nMaxTrackInConfirmedTrackTree);
	vecBestParamIndicator_[6].size = psn::size(nMaxTrackInUnconfirmedTrackTree);
	vecBestParamIndicator_[7].size = psn::size(nNumFrameForConfirmation);
	vecBestParamIndicator_[8].size = psn::size(fMaxMovingSpeed);
	vecBestParamIndicator_[9].size = psn::size(fProbEnterMax);
	vecBestParamIndicator_[10].size = psn::size(fProbEnterDecayCoef);
	vecBestParamIndicator_[11].size = psn::size(fCostEnterMax);
	vecBestParamIndicator_[12].size = psn::size(fProbExitMax);
	vecBestParamIndicator_[13].size = psn::size(fProbExitDecayCoef_dist);
	vecBestParamIndicator_[14].size = psn::size(fProbExitDecayCoef_length);
	vecBestParamIndicator_[15].size = psn::size(fCostExitMax);

	nTargetDataSet_ = DATASET_ETRI;
	rectCropZone_.x      = CROP_ZONE[nTargetDataSet_][0];
	rectCropZone_.y      = CROP_ZONE[nTargetDataSet_][2];
	rectCropZone_.width  = CROP_ZONE[nTargetDataSet_][1] - CROP_ZONE[nTargetDataSet_][0] + 1;
	rectCropZone_.height = CROP_ZONE[nTargetDataSet_][3] - CROP_ZONE[nTargetDataSet_][2] + 1;
	fCropZoneMargin_     = CROP_ZONE[nTargetDataSet_][4];

	bestParams_ = GetParameterFromIndicator(vecBestParamIndicator_);

	//paramSearchRecord bestRecord;
	//bestRecord.first = bestParams_;
	//vecSearchCache_.push_back(bestRecord);
}

CParameterOptimizer::~CParameterOptimizer(void)
{
}

AlgorithmParams CParameterOptimizer::Run()
{
	AlgorithmParams curParams;
	stParamRunInput curInput = {nTargetDataSet_, 0};
	
	curParams.params2DT.cropZone_       = rectCropZone_;
	curParams.params2DT.cropZoneMargin_ = fCropZoneMargin_;

	paramSearchRecord curRecord;
	curRecord.first = curParams;
	vecSearchCache_.push_back(curRecord);

	LocalSearch(&vecSearchCache_.back(), MIN_NUM_EVALUATION);

	return bestParams_;
}

bool CParameterOptimizer::RunAlgorithm(stParamRunInput runInput, AlgorithmParams params, stEvaluationResult &evaluationResult)
{
	char strInputFilePath[128];
	DATASET_TYPE dataset = runInput.datasetType;
	std::vector<cv::Mat> inputFrame(DATASET_NUM_CAM[dataset]);

	/////////////////////////////////////////////////////////////////
	// INITIALIZATION
	/////////////////////////////////////////////////////////////////		
	std::srand(runInput.seedValue);
	CPSNWhere psnWhere = CPSNWhere();
	psnWhere.Initialize(dataset, &params);

	/////////////////////////////////////////////////////////////////
	// MAIN LOOP
	/////////////////////////////////////////////////////////////////
	for (int frameIdx = DATASET_START_FRAME_IDX[dataset]; frameIdx <= DATASET_END_FRAME_IDX[dataset]; frameIdx++)
	{
		//---------------------------------------------------
		// FRAME GRABBING
		//---------------------------------------------------		
		for (int camIdx = 0; camIdx < DATASET_NUM_CAM[dataset]; camIdx++) 
		{
			unsigned int curCamID = DATASET_CAM_ID[dataset][camIdx];
			sprintf_s(strInputFilePath, sizeof(strInputFilePath), "%s\\View_%03d\\frame_%04d.jpg", DATASET_PATH[dataset].c_str(), curCamID, frameIdx);
			inputFrame[camIdx] = cv::imread(strInputFilePath, cv::IMREAD_COLOR);
			if (!inputFrame[camIdx].data)
			{
				std::cout << "Can't open the input frame" << std::endl;
				inputFrame[camIdx].release();
				return false;
			}
		}

		//---------------------------------------------------
		// TRACKING
		//---------------------------------------------------
		psnWhere.TrackPeople(inputFrame, frameIdx);

		//---------------------------------------------------
		// MEMORY CLEARING
		//---------------------------------------------------
		for (int camIdx = 0; camIdx < NUM_CAM; camIdx++) { inputFrame[camIdx].release(); }
	}

	/////////////////////////////////////////////////////////////////
	// TERMINATION
	/////////////////////////////////////////////////////////////////
	psnWhere.Finalize();

	return true;
}

AlgorithmParams CParameterOptimizer::LocalSearch(paramSearchRecord *startingParam, int minNumExecution)
{
	//---------------------------------------------------
	// GET MINIMUM RUN RESULTS FOR INITIAL POSITION
	//---------------------------------------------------
	while (minNumExecution > startingParam->second.size())
	{
		stParamRunInput    curRunInput = {nTargetDataSet_, (int)startingParam->second.size()};
		stParamOutputInfo  curOutputInfo;
		stEvaluationResult curEvalResult;
		RunAlgorithm(curRunInput, startingParam->first, curEvalResult);
		curOutputInfo.score = GetPerformance(curEvalResult);

		startingParam->second.push_back(std::make_pair(curRunInput, curOutputInfo));
	}
	paramSearchRecord *curBestParamRecord = startingParam;
	double bestScore = GetEstimatedPerformance(curBestParamRecord);
	vecBestParamIndicator_ = GetParameterIndicator(&curBestParamRecord->first);
	
	bool bLocalOptimum = false;
	while (!bLocalOptimum)
	{
		//---------------------------------------------------
		// FIND NEIGHBOR PARAMETERS
		//---------------------------------------------------
		std::deque<ParameterIndicatorSet> neighborIndicatorSets;
		for (int pIdx = 0; pIdx < NUM_PARAMS; pIdx++)
		{
			// parameter fixation
			if (std::end(fixedParameter) != std::find(std::begin(fixedParameter), std::end(fixedParameter), pIdx))
			{ continue; }

			if (0 < vecBestParamIndicator_[pIdx].position)
			{
				neighborIndicatorSets.push_back(vecBestParamIndicator_);
				neighborIndicatorSets.back()[pIdx].position--;
			}
			if (vecBestParamIndicator_[pIdx].position + 1 < vecBestParamIndicator_[pIdx].size)
			{
				neighborIndicatorSets.push_back(vecBestParamIndicator_);
				neighborIndicatorSets.back()[pIdx].position++;
			}			
		}

		//---------------------------------------------------
		// FIND NEIGHBOR CACHE RECORDS
		//---------------------------------------------------
		std::vector<paramSearchRecord*> ptVecParamSearchRecord(neighborIndicatorSets.size(), NULL);
		for (int nIdx = 0; nIdx < neighborIndicatorSets.size(); nIdx++)
		{
			AlgorithmParams paramsFromIndicator = GetParameterFromIndicator(neighborIndicatorSets[nIdx]);
			// search in the queue
			std::list<paramSearchRecord>::iterator findIter = std::find_if(vecSearchCache_.begin(), vecSearchCache_.end(), cacheSearch(paramsFromIndicator));
			if (vecSearchCache_.end() != findIter)
			{
				ptVecParamSearchRecord[nIdx] = &(*findIter);
			}
			else
			{
				paramSearchRecord newRecord;
				newRecord.first = paramsFromIndicator;
				newRecord.second.clear();
				vecSearchCache_.push_back(newRecord);
				ptVecParamSearchRecord[nIdx] = &vecSearchCache_.back();
			}
		}

		//---------------------------------------------------
		// EVALUATE NEIGHBOR PARAMETERS
		//---------------------------------------------------
		bool bMove = false;
		paramSearchRecord *maxParameterRecord = curBestParamRecord;
		for (int nIdx = 0; nIdx < ptVecParamSearchRecord.size(); nIdx++)
		{			
			while (maxParameterRecord->second.size() > ptVecParamSearchRecord[nIdx]->second.size())
			{
				stParamRunInput    curRunInput = {nTargetDataSet_, (int)ptVecParamSearchRecord[nIdx]->second.size()};
				stParamOutputInfo  curOutputInfo;
				stEvaluationResult curEvalResult;
				RunAlgorithm(curRunInput, ptVecParamSearchRecord[nIdx]->first, curEvalResult);
				curOutputInfo.score = GetPerformance(curEvalResult);
				ptVecParamSearchRecord[nIdx]->second.push_back(std::make_pair(curRunInput, curOutputInfo));

				// early quit
				if (bestScore > GetPossibleBestPerformance(ptVecParamSearchRecord[nIdx], (int)maxParameterRecord->second.size()))
				{
					break;
				}
			}

			double curEstimatedScore = GetEstimatedPerformance(ptVecParamSearchRecord[nIdx]);
			if (bestScore < curEstimatedScore)
			{
				bestScore = curEstimatedScore;
				maxParameterRecord = ptVecParamSearchRecord[nIdx];
				bMove = true;
			}
		}
		if (!bMove) { bLocalOptimum = true; }
	}
	
	return curBestParamRecord->first;
}

//AlgorithmParams CParameterOptimizer::Perturbation(void)
//{
//
//}

double CParameterOptimizer::GetPerformance(stEvaluationResult &result)
{
	return result.fMOTA;
}

double CParameterOptimizer::GetEstimatedPerformance(paramSearchRecord *records, int numEvaluations)
{
	
	double totalScore = 0.0;
	int numRecords = (int)records->second.size();
	if (0 >= numRecords) { return -DBL_MAX; }
	if (0 < numEvaluations)
	{
		if (numEvaluations > numRecords) { printf("[WARNING] insufficient number of records to compare\n"); }
		numRecords = std::min(numRecords, numEvaluations);
	}

	// get median value
	std::vector<double> scores(numRecords, 0.0);
	for (int i = 0; i < numRecords; i++)
	{
		scores[i] = records->second[i].second.score;
	}
	return psn::Median(scores);
}

double CParameterOptimizer::GetPossibleBestPerformance(paramSearchRecord *records, int numTotalEvaluations)
{
	// median value
	if (numTotalEvaluations / 2 > records->second.size()) { return 1.0; }
	std::vector<double> scores(numTotalEvaluations, 1.0);
	for (int i = 0; i < records->second.size(); i++)
	{
		scores[i] = records->second[i].second.score;
	}
	return psn::Median(scores);
}

ParameterIndicatorSet CParameterOptimizer::GetParameterIndicator(AlgorithmParams *parameter)
{
	const double *p0 = std::find(std::begin(nBackTrackingInterval), std::end(nBackTrackingInterval), parameter->params2DT.nBackTrackingInterval_);
	assert(p0 != nBackTrackingInterval + psn::size(nBackTrackingInterval));
	const double *p1 = std::find(std::begin(fFeatureWindowSizeRatio), std::end(fFeatureWindowSizeRatio), parameter->params2DT.fFeatureWindowSizeRatio_);
	assert(p1 != fFeatureWindowSizeRatio + psn::size(fFeatureWindowSizeRatio));

	const double *p2 = std::find(std::begin(nProcWindowSize), std::end(nProcWindowSize), parameter->params3DA.nProcWindowSize_);
	assert(p2 != nProcWindowSize + psn::size(nProcWindowSize));
	const double *p3 = std::find(std::begin(nKBestSize), std::end(nKBestSize), parameter->params3DA.nKBestSize_);
	assert(p3 != nKBestSize + psn::size(nKBestSize));
	const double *p4 = std::find(std::begin(nMaxTrackInOptimization), std::end(nMaxTrackInOptimization), parameter->params3DA.nMaxTrackInOptimization_);
	assert(p4 != nMaxTrackInOptimization + psn::size(nMaxTrackInOptimization));
	const double *p5 = std::find(std::begin(nMaxTrackInConfirmedTrackTree), std::end(nMaxTrackInConfirmedTrackTree), parameter->params3DA.nMaxTrackInConfirmedTrackTree_);
	assert(p5 != nMaxTrackInConfirmedTrackTree + psn::size(nMaxTrackInConfirmedTrackTree));
	const double *p6 = std::find(std::begin(nMaxTrackInUnconfirmedTrackTree), std::end(nMaxTrackInUnconfirmedTrackTree), parameter->params3DA.nMaxTrackInUnconfirmedTrackTree_);
	assert(p6 != nMaxTrackInUnconfirmedTrackTree + psn::size(nMaxTrackInUnconfirmedTrackTree));
	const double *p7 = std::find(std::begin(nNumFrameForConfirmation), std::end(nNumFrameForConfirmation), parameter->params3DA.nNumFrameForConfirmation_);
	assert(p7 != nNumFrameForConfirmation + psn::size(nNumFrameForConfirmation));

	const double *p8 = std::find(std::begin(fMaxMovingSpeed), std::end(fMaxMovingSpeed), parameter->params3DA.fMaxMovingSpeed_);
	assert(p8 != fMaxMovingSpeed + psn::size(fMaxMovingSpeed));

	const double *p9 = std::find(std::begin(fProbEnterMax), std::end(fProbEnterMax), parameter->params3DA.fProbEnterMax_);
	assert(p9 != fProbEnterMax + psn::size(fProbEnterMax));
	const double *p10 = std::find(std::begin(fProbEnterDecayCoef), std::end(fProbEnterDecayCoef), parameter->params3DA.fProbEnterDecayCoef_);
	assert(p10 != fProbEnterDecayCoef + psn::size(fProbEnterDecayCoef));
	const double *p11 = std::find(std::begin(fCostEnterMax), std::end(fCostEnterMax), parameter->params3DA.fCostEnterMax_);
	assert(p11 != fCostEnterMax + psn::size(fCostEnterMax));

	const double *p12 = std::find(std::begin(fProbExitMax), std::end(fProbExitMax), parameter->params3DA.fProbExitMax_);
	assert(p12 != fProbExitMax + psn::size(fProbExitMax));
	const double *p13 = std::find(std::begin(fProbExitDecayCoef_dist), std::end(fProbExitDecayCoef_dist), parameter->params3DA.fProbExitDecayCoef_dist_);
	assert(p13 != fProbExitDecayCoef_dist + psn::size(fProbExitDecayCoef_dist));
	const double *p14 = std::find(std::begin(fProbExitDecayCoef_length), std::end(fProbExitDecayCoef_length), parameter->params3DA.fProbExitDecayCoef_length_);
	assert(p14 != fProbExitDecayCoef_length + psn::size(fProbExitDecayCoef_length));
	const double *p15 = std::find(std::begin(fCostExitMax), std::end(fCostExitMax), parameter->params3DA.fCostExitMax_);
	assert(p15 != fCostExitMax + psn::size(fCostExitMax));

	std::vector<stParameterIndicator> curIndicator(NUM_PARAMS);
	curIndicator[0].position = (int)(p0 - nBackTrackingInterval);
	curIndicator[0].size     = psn::size(nBackTrackingInterval);
	curIndicator[1].position = (int)(p1 - fFeatureWindowSizeRatio);
	curIndicator[1].size     = psn::size(fFeatureWindowSizeRatio);

	curIndicator[2].position = (int)(p2 - nProcWindowSize);
	curIndicator[2].size     = psn::size(nProcWindowSize);
	curIndicator[3].position = (int)(p3 - nKBestSize);
	curIndicator[3].size     = psn::size(nKBestSize);
	curIndicator[4].position = (int)(p4 - nMaxTrackInOptimization);
	curIndicator[4].size     = psn::size(nMaxTrackInOptimization);
	curIndicator[5].position = (int)(p5 - nMaxTrackInConfirmedTrackTree);
	curIndicator[5].size     = psn::size(nMaxTrackInConfirmedTrackTree);
	curIndicator[6].position = (int)(p6 - nMaxTrackInUnconfirmedTrackTree);
	curIndicator[6].size     = psn::size(nMaxTrackInUnconfirmedTrackTree);
	curIndicator[7].position = (int)(p7 - nNumFrameForConfirmation);
	curIndicator[7].size     = psn::size(nNumFrameForConfirmation);

	curIndicator[8].position = (int)(p8 - fMaxMovingSpeed);
	curIndicator[8].size     = psn::size(fMaxMovingSpeed);

	curIndicator[9].position = (int)(p9 - fProbEnterMax);
	curIndicator[9].size     = psn::size(fProbEnterMax);
	curIndicator[10].position = (int)(p10 - fProbEnterDecayCoef);
	curIndicator[10].size     = psn::size(fProbEnterDecayCoef);
	curIndicator[11].position = (int)(p11 - fCostEnterMax);
	curIndicator[11].size     = psn::size(fCostEnterMax);

	curIndicator[12].position = (int)(p12 - fProbExitMax);
	curIndicator[12].size     = psn::size(fProbExitMax);
	curIndicator[13].position = (int)(p13 - fProbExitDecayCoef_dist);
	curIndicator[13].size     = psn::size(fProbExitDecayCoef_dist);
	curIndicator[14].position = (int)(p14 - fProbExitDecayCoef_length);
	curIndicator[14].size     = psn::size(fProbExitDecayCoef_length);
	curIndicator[15].position = (int)(p15 - fCostExitMax);
	curIndicator[15].size     = psn::size(fCostExitMax);

	return curIndicator;
}


AlgorithmParams CParameterOptimizer::GetParameterFromIndicator(ParameterIndicatorSet &indicatorSet)
{
	AlgorithmParams parameters;
	parameters.params2DT.nBackTrackingInterval_   = (int)paramDomain[0][indicatorSet[0].position];
	parameters.params2DT.fFeatureWindowSizeRatio_ = paramDomain[1][indicatorSet[1].position];
	parameters.params2DT.cropZone_                = rectCropZone_;
	parameters.params2DT.cropZoneMargin_          = fCropZoneMargin_;
		
	for (int pIdx = 2; pIdx < NUM_PARAMS; pIdx++)
	{
		parameters.params3DA.SetParameter(pIdx - 2, paramDomain[pIdx][indicatorSet[pIdx].position]);
	}
	return parameters;
}


