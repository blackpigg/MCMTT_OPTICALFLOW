#include "ParameterOptimizer.h"



// 2D tracking parameter domains
const int nBackTrackingInterval[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const double fFeatureClusterRadiusRatio[] = {0.1, 0.2, 0.3, 0.5, 0.7, 1.0};

// 3D association parameter domains
const int nProcWindowSize[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
const int nKBestSize[] = {1, 5, 10, 15, 20, 25, 30};
const int nMaxTrackInOptimization[] = {100, 500, 1000, 1500, 2000, 2500, 3000};
const int nMaxTrackInConfirmedTrackTree[] = {10, 50, 100, 200, 300};
const int nMaxTrackInUnconfirmedTrackTree[] = {2, 3, 4, 5, 10};
const int nNumFrameForConfirmation[] = {1, 2, 3, 4, 5};

const double fMaxMovingSpeed[] = {500, 900, 1000, 2000};

const double fProbEnterMax[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fProbEnterDecayCoef[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fCostEnterMax[] = {100, 200, 500, 1000, 1500};

const double fProbExitMax[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fProbExitDecayCoef_dist[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fProbExitDecayCoef_length[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6};
const double fCostExitMax[] = {100, 200, 500, 1000, 1500};


CParameterOptimizer::CParameterOptimizer(void)
{
}


CParameterOptimizer::~CParameterOptimizer(void)
{
}

AlgorithmParams CParameterOptimizer::Run()
{
	nTargetDataSet_ = DATASET_PETS2009_S2L3;


}

stEvaluationResult CParameterOptimizer::RunAlgorithm(DATASET_TYPE datasetType, AlgorithmParams params)
{
	char    strInputFilePath[128];
	cv::Mat inputFrame[DATASET_NUM_CAM[datasetType]];

	/////////////////////////////////////////////////////////////////
	// INITIALIZATION
	/////////////////////////////////////////////////////////////////		
	CPSNWhere psnWhere = CPSNWhere();
	psnWhere.Initialize(DATASET_PATH[datasetType], &paramsAssociator3D);

	/////////////////////////////////////////////////////////////////
	// MAIN LOOP
	/////////////////////////////////////////////////////////////////
	for (int frameIdx = DATASET_START_FRAME_IDX[datasetType]; frameIdx <= DATASET_END_FRAME_IDX[datasetType]; frameIdx++)
	{
		//---------------------------------------------------
		// FRAME GRABBING
		//---------------------------------------------------		
		for (int camIdx = 0; camIdx < DATASET_NUM_CAM[datasetType]; camIdx++) 
		{
			unsigned int curCamID = DATASET_CAM_ID[datasetType][camIdx];
			sprintf_s(strInputFilePath, sizeof(strInputFilePath), "%s\\View_%03d\\frame_%04d.jpg", DATASET_PATH[datasetType].c_str(), curCamID, frameIdx);
		}
	}
}
