/*************************************************************************************
NOTES:
 - There are some inaccurate calculations in 3D point reconstruction and vision prior
 at branching stage for conveniency for computing.
**************************************************************************************/
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include "PSNWhere_Associator3D.h"
#include "Helpers\PSNWhere_Hungarian.h"

//#define PSN_DEBUG_MODE
//#define PSN_PRINT_LOG

/////////////////////////////////////////////////////////////////////////
// PARAMETERS (unit: mm)
/////////////////////////////////////////////////////////////////////////

// optimization related
#define PROC_WINDOW_SIZE (20)
#define GTP_THRESHOLD (0.0001)
#define MAX_TRACK_IN_OPTIMIZATION (1000)
#define MAX_UNCONFIRMED_TRACK (50)
#define NUM_FRAME_FOR_GTP_CHECK (3)
#define NUM_FRAME_FOR_CONFIRMATION (5)
#define K_BEST_SIZE (50)
#define DO_BRANCH_CUT (false)

// reconstruction related
#define MIN_TRACKLET_LENGTH (1)
#define MAX_TRACKLET_LENGTH (15)
#define MAX_TRACKLET_DISTANCE (3000)
#define MAX_TRACKLET_SENSITIVITY_ERROR (20)

#define MAX_HEAD_WIDTH (420)
#define MIN_HEAD_WIDTH (120)
#define MAX_HEAD_HEIGHT (2300)
#define MIN_HEAD_HEIGHT (0)
#define HEAD_WIDTH_MEAN (245.7138)
#define HEAD_WIDTH_STD (25.4697)

#define MAX_BODY_WIDHT (MAX_TRACKLET_DISTANCE)
#define BODY_WIDTH_MEAN (700)
#define BODY_WIDTH_STD (100)
#define MAX_TARGET_PROXIMITY (500)

#define DEFAULT_HEIGHT (1700.00)
#define WIDTH_DIFF_RATIO_THRESHOLD (0.5)

// linking related
#define MIN_LINKING_PROBABILITY (1.0E-6)
#define MAX_TIME_JUMP (9)

#define MAX_GROUNDING_HEIGHT (100)

// probability related
#define MIN_CONSTRUCT_PROBABILITY (0.01)
#define FP_RATE (0.05)
#define FN_RATE (0.1)

// enter/exit related
#define ENTER_PENALTY_FREE_LENGTH (3)
#define BOUNDARY_DISTANCE (1000.0)
#define P_EN_MAX (0.1)
#define P_EX_MAX (1.0E-6)
#define P_EN_DECAY (1/100)
#define P_EX_DECAY (1/10)
#define COST_EN_MAX (300.0)
#define COST_EX_MAX (300.0)

// calibration related
#define E_DET (4)
#define E_CAL (500)

// dynamic related
#define KALMAN_PROCESSNOISE_SIG (1.0E-5)
#define KALMAN_MEASUREMENTNOISE_SIG (1.0E-5)
#define KALMAN_POSTERROR_COV (0.1)
#define KALMAN_CONFIDENCE_LEVEN (9)
#define VELOCITY_LEARNING_RATE (0.9)
//#define MAX_MOVING_SPEED (5000.0/DATASET_FRAME_RATE)
#define DATASET_FRAME_RATE (7)
#define MAX_MOVING_SPEED (800)

// appearance reltaed
#define IMG_PATCH_WIDTH (20)

// ETC
//const unsigned int arrayCamCombination[8] = {0, 2, 2, 1, 1, 3, 3, 0};


/////////////////////////////////////////////////////////////////////////
// LOCAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////

// compartors for sorting
bool psnTrackIDAscendComparator(const Track3D *track1, const Track3D *track2)
{
	return track1->id < track2->id;
}

bool psnTrackAscendentCostComparator(const Track3D *track1, const Track3D *track2)
{
	return track1->costTotal < track2->costTotal;
}

bool psnTrackGPDescendComparator(const Track3D *track1, const Track3D *track2)
{	
	return track1->GTProb > track2->GTProb;
}

bool psnTrackGPDescendAndDurationAscendComparator(const Track3D *track1, const Track3D *track2)
{
	// higher GTProb, shorter duration!
	if(track1->GTProb > track2->GTProb)
	{
		return true;
	}
	else if(track1->GTProb == track2->GTProb)
	{
		if(track1->duration < track2->duration)
		{
			return true;
		}
	}
	return false;
}

bool psnTrackGTPandLLDescend(const Track3D *track1, const Track3D *track2)
{
	// higher GTProb, lower cost!
	if(track1->GTProb > track2->GTProb)
	{
		return true;
	}
	else if(track1->GTProb == track2->GTProb)
	{
		if(track1->loglikelihood > track2->loglikelihood)
		{
			return true;
		}
	}
	return false;
}

bool psnTrackGPandLengthComparator(const Track3D *track1, const Track3D *track2)
{
	if(track1->duration > NUM_FRAME_FOR_CONFIRMATION && track2->duration > NUM_FRAME_FOR_CONFIRMATION)
	{
		if(track1->GTProb > track2->GTProb)
		{
			return true;
		}
		else if(track1->GTProb < track2->GTProb)
		{
			return false;
		}

		if(track1->duration > track2->duration)
		{
			return true;
		}
		return false;
	}
	else if(track1->duration >= NUM_FRAME_FOR_CONFIRMATION && track2->duration < NUM_FRAME_FOR_CONFIRMATION)
	{
		return false;
	}
	return true;
}

bool psnTreeNumMeasurementDescendComparator(const TrackTree *tree1, const TrackTree *tree2)
{
	return tree1->numMeasurements > tree2->numMeasurements;
}

bool psnSolutionLogLikelihoodDescendComparator(const stGlobalHypothesis &solution1, const stGlobalHypothesis &solution2)
{
	return solution1.logLikelihood > solution2.logLikelihood;
}

bool psnUnconfirmedTrackGTPLengthDescendComparator(const TrackTree *tree1, const TrackTree *tree2)
{
	if(tree1->maxGTProb > tree1->maxGTProb)
	{
		return true;
	}	
	if(tree1->maxGTProb < tree1->maxGTProb)
	{
		return false;
	}
	if(tree1->timeGeneration < tree2->timeGeneration)
	{
		return true;
	}
	return false;
}


/////////////////////////////////////////////////////////////////////////
// MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////


/************************************************************************
 Method Name: CPSNWhere_Associator3D
 Description: 
	- class constructor
 Input Arguments:
	-
	-
 Return Values:
	- class instance
************************************************************************/
CPSNWhere_Associator3D::CPSNWhere_Associator3D(void)
	: bInit_(false)
{
	//for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	//{
	//	cCamModel_[camIdx] = NULL;
	//}
}


/************************************************************************
 Method Name: ~CPSNWhere_Associator3D
 Description: 
	- class destructor
 Input Arguments:
	-
	-
 Return Values:
	- none
************************************************************************/
CPSNWhere_Associator3D::~CPSNWhere_Associator3D(void)
{
}


/************************************************************************
 Method Name: Initialize
 Description: 
	- initialization routine for 3D association
 Input Arguments:
	- datasetPath: the string for a dataset path
	- stCalibInfo: calibration information
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Initialize(std::string datasetPath, std::vector<stCalibrationInfo*> &vecStCalibInfo)
{
#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](Initialize) start\n");
#endif
	if (bInit_) { return; }

	sprintf_s(strDatasetPath_, "%s", datasetPath.c_str());
	nCurrentFrameIdx_ = 0;
	nNumFramesForProc_ = 0;
	nCountForPenalty_ = 0;
	fCurrentProcessingTime_ = 0.0;
	fCurrentSolvingTime_ = 0.0;

	// 2D tracklet related
	nNumTotalActive2DTracklet_ = 0;

	// 3D track related
	nNewTrackID_ = 0;
	nNewTreeID_ = 0;
	nNewHypothesisID = 0;
	nLastPrintedDeferredResultFrameIdx_ = 0;
	nLastPrintedInstantResultFrameIdx_ = 0;
	bReceiveNewMeasurement_ = false;
	bInitiationPenaltyFree_ = true;

	queueNewSeedTracks_.clear();
	queueActiveTrack_.clear();
	queuePausedTrack_.clear();
	queueTracksInWindow_.clear();
	queueTracksInBestSolution_.clear();

	// optimization related
	queuePrevGlobalHypotheses_.clear();
	queueCurrGlobalHypotheses_.clear();

	// visualiztion related
	nNewVisualizationID_ = 0;
	queuePairTreeIDToVisualizationID_.clear();

	// camera model
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		cCamModel_[camIdx] = vecStCalibInfo[camIdx]->cCamModel;
		matProjectionSensitivity_[camIdx] = vecStCalibInfo[camIdx]->matProjectionSensitivity.clone();
		matDistanceFromBoundary_[camIdx] = vecStCalibInfo[camIdx]->matDistanceFromBoundary.clone();
	}
	
	//// evaluation
	//this->m_cEvaluator.Initialize(datasetPath);
	//this->m_cEvaluator_Instance.Initialize(datasetPath);


	// logging related
	time_t curTimer = time(NULL);
	struct tm timeStruct;
	localtime_s(&timeStruct, &curTimer);
	sprintf_s(strLogFileName_, "logs/PSN_Log_%02d%02d%02d_%02d%02d%02d.txt", 
		timeStruct.tm_year + 1900, 
		timeStruct.tm_mon+1, 
		timeStruct.tm_mday, 
		timeStruct.tm_hour, 
		timeStruct.tm_min, 
		timeStruct.tm_sec);

	sprintf_s(strTrackLogFileName_, "%stracks/PSN_tracks_%02d%02d%02d_%02d%02d%02d.txt",
		RESULT_SAVE_PATH,
		timeStruct.tm_year + 1900, 
		timeStruct.tm_mon+1, 
		timeStruct.tm_mday, 
		timeStruct.tm_hour, 
		timeStruct.tm_min, 
		timeStruct.tm_sec);

	sprintf_s(strResultLogFileName_, "%stracking/PSN_BatchResult_%04d%02d%02d%02d%02d%02d.txt", 
		RESULT_SAVE_PATH,
		timeStruct.tm_year + 1900, 
		timeStruct.tm_mon+1, 
		timeStruct.tm_mday, 
		timeStruct.tm_hour, 
		timeStruct.tm_min, 
		timeStruct.tm_sec);

	sprintf_s(strDefferedResultFileName_, "%stracking/PSN_DefferedResult_%04d%02d%02d%02d%02d%02d.txt", 
		RESULT_SAVE_PATH,
		timeStruct.tm_year + 1900, 
		timeStruct.tm_mon+1, 
		timeStruct.tm_mday, 
		timeStruct.tm_hour, 
		timeStruct.tm_min, 
		timeStruct.tm_sec);

	sprintf_s(strInstantResultFileName_, "%stracking/PSN_CurrentResult_%04d%02d%02d%02d%02d%02d.txt", 
		RESULT_SAVE_PATH,
		timeStruct.tm_year + 1900, 
		timeStruct.tm_mon+1, 
		timeStruct.tm_mday, 
		timeStruct.tm_hour, 
		timeStruct.tm_min, 
		timeStruct.tm_sec);

	bInit_ = true;

#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](Initialize) end\n");
#endif
}


/************************************************************************
 Method Name: Finalize
 Description: 
	- termination routine for 3D association
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Finalize(void)
{
#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](Finalize) start\n");
#endif

	if(!bInit_){ return; }

	/////////////////////////////////////////////////////////////////////////////
	// PRINT RESULT
	/////////////////////////////////////////////////////////////////////////////
#ifdef PSN_PRINT_LOG
	//// candidate tracks
	//this->PrintTracks(queuePausedTrack_, strTrackLogFileName_, true);
	//this->PrintTracks(queueActiveTrack_, strTrackLogFileName_, true);
#endif

	// tracks in solutions
	std::deque<Track3D*> queueResultTracks;
	for(std::list<Track3D>::iterator trackIter = listResultTrack3D_.begin();
		trackIter != listResultTrack3D_.end();
		trackIter++)
	{
		queueResultTracks.push_back(&(*trackIter));
	}
	for(std::deque<Track3D*>::iterator trackIter = queueTracksInBestSolution_.begin();
		trackIter != queueTracksInBestSolution_.end();
		trackIter++)
	{
		queueResultTracks.push_back(*trackIter);
	}


#ifdef PSN_PRINT_LOG
	// track print
	PSN_TrackSet allTracks;
	for(std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
		trackIter != listTrack3D_.end();
		trackIter++)
	{
		allTracks.push_back(&(*trackIter));
	}	
	this->PrintTracks(allTracks, strTrackLogFileName_, false);
	//this->PrintTracks(queueResultTracks, strResultLogFileName_, false);

	//// deferred result
	//this->SaveDefferedResult(0);
	//this->FilePrintDefferedResult();
	//this->FilePrintInstantResult();
#endif
	// print results
	this->SaveDefferedResult(0);
	this->FilePrintDefferedResult();
	this->FilePrintInstantResult();

	///////////////////////////////////////////////////////////////////////////////
	//// EVALUATION
	///////////////////////////////////////////////////////////////////////////////
	//printf("[EVALUATION] deferred result\n");
	//for(unsigned int timeIdx = nCurrentFrameIdx_ - PROC_WINDOW_SIZE + 2; timeIdx <= nCurrentFrameIdx_; timeIdx++)
	//{
	//	this->m_cEvaluator.SetResult(queueStGraphSolutions_[0].tracks, timeIdx);
	//}
	//this->m_cEvaluator.Evaluate();
	//this->m_cEvaluator.PrintResultToConsole();

	//printf("[EVALUATION] instance result\n");
	//this->m_cEvaluator_Instance.Evaluate();
	//this->m_cEvaluator.PrintResultToConsole();


	/////////////////////////////////////////////////////////////////////////////
	// FINALIZE
	/////////////////////////////////////////////////////////////////////////////

	// clean-up camera model
	for(int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		//delete cCamModel_[camIdx];
		matProjectionSensitivity_[camIdx].release();
		matDistanceFromBoundary_[camIdx].release();
	}

	fCurrentProcessingTime_ = 0.0;
	nCurrentFrameIdx_ = 0;

	// 2D tracklet related
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		vecTracklet2DSet_[camIdx].activeTracklets.clear();
		for(std::list<stTracklet2D>::iterator trackletIter = vecTracklet2DSet_[camIdx].tracklets.begin();
			trackletIter != vecTracklet2DSet_[camIdx].tracklets.end();
			trackletIter++)
		{
			trackletIter->rects.clear();
		}
		vecTracklet2DSet_[camIdx].tracklets.clear();
	}

	// 3D track related
	nNewTrackID_ = 0;
	nNewTreeID_ = 0;

#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](Finalize) end\n");
#endif
}


/************************************************************************
 Method Name: Run
 Description: 
	- main process for 3D association
 Input Arguments:
	-
	-
 Return Values:
	- stTrack3DResult: 3D tracking result from the processing
************************************************************************/
stTrack3DResult CPSNWhere_Associator3D::Run(std::vector<stTrack2DResult> &curTrack2DResult, cv::Mat *curFrame, int frameIdx)
{
#ifdef PSN_DEBUG_MODE_
	printf("==================================================\n", frameIdx);
	printf("[CPSNWhere_Associator3D](Run) start with frame: %d\n", frameIdx);
#endif
	assert(bInit_);

	clock_t timer_start;
	clock_t timer_end;
	timer_start = clock();
	double processingTime;

	/////////////////////////////////////////////////////////////////////////////
	// PRE-PROCESSING
	/////////////////////////////////////////////////////////////////////////////
	// get frames	
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++) { ptMatCurrentFrames_[camIdx] = &curFrame[camIdx]; }
	nCurrentFrameIdx_ = frameIdx;
	nNumFramesForProc_++;
	// enterance/exit penalty
	if (bInitiationPenaltyFree_)
	{
		if (nCountForPenalty_ > ENTER_PENALTY_FREE_LENGTH) { bInitiationPenaltyFree_ = false; }
		nCountForPenalty_++;
	}

	/////////////////////////////////////////////////////////////////////////////
	// 2D TRACKLET
	/////////////////////////////////////////////////////////////////////////////
	this->Tracklet2D_UpdateTracklets(curTrack2DResult, frameIdx);

	/////////////////////////////////////////////////////////////////////////////
	// 3D TRACK
	/////////////////////////////////////////////////////////////////////////////	
	this->Track3D_Management(queueNewSeedTracks_);

	/////////////////////////////////////////////////////////////////////////////
	// GLOBAL HYPOTHESES
	/////////////////////////////////////////////////////////////////////////////	
	// solve MHT
	this->Hypothesis_UpdateHypotheses(queuePrevGlobalHypotheses_, &queueNewSeedTracks_);
	this->Hypothesis_Formation(queueCurrGlobalHypotheses_, &queuePrevGlobalHypotheses_);	
	//this->Track3D_SolveHOMHT();

	// post-pruning
	this->Hypothesis_PruningNScanBack(nCurrentFrameIdx_, PROC_WINDOW_SIZE, &queueTracksInWindow_, &queueCurrGlobalHypotheses_);

	//this->Track3D_Pruning_KBest();
	//if(MAX_TRACK_IN_OPTIMIZATION < queueTracksInWindow_.size())
	//{
	//	this->Track3D_Pruning_KBest();
	//}

	/////////////////////////////////////////////////////////////////////////////
	// RESULT PACKING
	/////////////////////////////////////////////////////////////////////////////

	// measuring processing time
	timer_end = clock();
	processingTime = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
	fCurrentProcessingTime_ = processingTime;

	// packing current tracking result
	if (0 < queueCurrGlobalHypotheses_.size()) 
	{
		queueTracksInBestSolution_ = queueCurrGlobalHypotheses_.front().selectedTracks;
	} 
	else 
	{
		queueTracksInBestSolution_.clear();
	}
	stTrack3DResult currentResult = this->ResultWithTracks(&queueTracksInBestSolution_, frameIdx, processingTime); 
	this->SaveInstantResult();
	this->SaveDefferedResult(PROC_WINDOW_SIZE);

	/////////////////////////////////////////////////////////////////////////////
	// WRAP-UP
	/////////////////////////////////////////////////////////////////////////////

	// memory clean-up
	for(int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		ptMatCurrentFrames_[camIdx] = NULL;
	}

	// hypothesis backup
	queuePrevGlobalHypotheses_ = queueCurrGlobalHypotheses_;
	queueCurrGlobalHypotheses_.clear();

#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](Run) processingTime:%f sec\n", fCurrentProcessingTime_);
	printf("[CPSNWhere_Associator3D](Run) total candidate tracks:%d\n", listTrack3D_.size());
#endif

#ifdef PSN_PRINT_LOG
	// PRINT LOG
	char strLog[128];
	sprintf_s(strLog, "%d,%d,%d,%f,%f", 
	nCurrentFrameIdx_, 
	queueActiveTrack_.size(),
	queueStGraphSolutions_.size(),
	fCurrentProcessingTime_,
	fCurrentSolvingTime_);
	CPSNWhere_Manager::printLog(strLogFileName_, strLog);
	fCurrentSolvingTime_ = 0.0;
#endif

	return currentResult;
}


/************************************************************************
 Method Name: GetHuman3DBox
 Description: 
	- find a corner points of 3D box for a given 3D point
 Input Arguments:
	- ptHeadCenter: 3D point of a center of the head
	- bodyWidth: a width of the box 
	- camIdx: an index of the camera for display
 Return Values:
	- std::vector<PSN_Point2D>: 3D tracking result from the processing
************************************************************************/
std::vector<PSN_Point2D> CPSNWhere_Associator3D::GetHuman3DBox(PSN_Point3D ptHeadCenter, double bodyWidth, unsigned int camIdx)
{
	std::vector<PSN_Point2D> resultArray;
	double bodyWidthHalf = bodyWidth / 2.0;

	double offsetX[8] = {+bodyWidthHalf, +bodyWidthHalf, -bodyWidthHalf, -bodyWidthHalf, +bodyWidthHalf, +bodyWidthHalf, -bodyWidthHalf, -bodyWidthHalf};
	double offsetY[8] = {-bodyWidthHalf, +bodyWidthHalf, +bodyWidthHalf, -bodyWidthHalf, -bodyWidthHalf, +bodyWidthHalf, +bodyWidthHalf, -bodyWidthHalf};
	double z[8] = {ptHeadCenter.z+bodyWidthHalf, ptHeadCenter.z+bodyWidthHalf, ptHeadCenter.z+bodyWidthHalf, ptHeadCenter.z+bodyWidthHalf, 0, 0, 0, 0};

	unsigned int numPointsOnView = 0;
	for(unsigned int vertexIdx = 0; vertexIdx < 8; vertexIdx++)
	{
		PSN_Point2D curPoint = this->WorldToImage(PSN_Point3D(ptHeadCenter.x + offsetX[vertexIdx], ptHeadCenter.y + offsetY[vertexIdx], z[vertexIdx]), camIdx);
		resultArray.push_back(curPoint);
		if(curPoint.onView(cCamModel_[camIdx].width(), cCamModel_[camIdx].height())){ numPointsOnView++; }
	}

	// if there is no point can be seen from the view, clear a vector of point
	if(0 == numPointsOnView)
	{
		resultArray.clear();
	}

	return resultArray;
}


/************************************************************************
 Method Name: WorldToImage
 Description: 
	- convert a world point to a point on the image (3D -> 2D)
 Input Arguments:
	- point3D: 3D point
	- camIdx: an index of target camera
 Return Values:
	- PSN_Point2D: image coordinate of the point (2D)
************************************************************************/
PSN_Point2D CPSNWhere_Associator3D::WorldToImage(PSN_Point3D point3D, int camIdx)
{
	PSN_Point2D resultPoint2D(0, 0);
	cCamModel_[camIdx].worldToImage(point3D.x, point3D.y, point3D.z, resultPoint2D.x, resultPoint2D.y);
	return resultPoint2D;
}


/************************************************************************
 Method Name: WorldToImage
 Description: 
	- convert an image point to 3D point with a specific height (2D -> 3D)
 Input Arguments:
	- point2D: a point on the image
	- z: height of reconstructed point
	- camIdx: an index of camera which a 2D point came from
 Return Values:
	- PSN_Point2D: image coordinate of the point (2D)
************************************************************************/
PSN_Point3D CPSNWhere_Associator3D::ImageToWorld(PSN_Point2D point2D, double z, int camIdx)
{
	PSN_Point3D resultPoint3D(0, 0, 0);
	resultPoint3D.z = z;
	cCamModel_[camIdx].imageToWorld(point2D.x, point2D.y, z, resultPoint3D.x, resultPoint3D.y);
	return resultPoint3D;
}


/************************************************************************
 Method Name: ReadProjectionSensitivity
 Description: 
	- read sensitivity of projection matrix
 Input Arguments:
	- matSensitivity: map of sensitivity of a projection matrix
	- camIdx: an index of camera which a projection matrix cames from
 Return Values:
	- whether file read successfully
************************************************************************/
bool CPSNWhere_Associator3D::ReadProjectionSensitivity(cv::Mat &matSensitivity, unsigned int camIdx)
{
	char strFilePath[128];
	sprintf_s(strFilePath, "%s%sProjectionSensitivity_View%03d.txt", 
		strDatasetPath_, 
		CALIBRATION_PATH, 
		CAM_ID[camIdx]);

	FILE *fp;
	int numRows = 0, numCols = 0;
	float curSensitivity = 0.0;

	try
	{
		fopen_s(&fp, strFilePath, "r");
		fscanf_s(fp, "row:%d,col:%d\n", &numRows, &numCols);
		matSensitivity = cv::Mat::zeros(numRows, numCols, CV_32FC1);

		for(unsigned int rowIdx = 0; rowIdx < (unsigned int)numRows; rowIdx++)
		{
			for(unsigned int colIdx = 0; colIdx < (unsigned int)numCols; colIdx++)
			{
				fscanf_s(fp, "%f,", &curSensitivity);
				matSensitivity.at<float>(rowIdx, colIdx) = curSensitivity;
			}
			fscanf_s(fp, "\n");
		}

		fclose(fp);
		return true;
	}
	catch(DWORD dwError)
	{
		printf("[ERROR] cannot open projection sensitivity matrix of camera %d! error code %d\n", CAM_ID[camIdx], dwError);
	}
	return false;
}

/************************************************************************
 Method Name: ReadDistanceFromBoundary
 Description: 
	- read matrix of distance from boundary
 Input Arguments:
	- matDistance: map of distance from boundary
	- camIdx: an index of camera which a projection matrix cames from
 Return Values:
	- whether file read successfully
************************************************************************/
bool CPSNWhere_Associator3D::ReadDistanceFromBoundary(cv::Mat &matDistance, unsigned int camIdx)
{
	char strFilePath[128];
	sprintf_s(strFilePath, "%s%sDistanceFromBoundary_View%03d.txt", 
		strDatasetPath_, 
		CALIBRATION_PATH, 
		CAM_ID[camIdx]);

	FILE *fp;
	int numRows = 0, numCols = 0;
	float curSensitivity = 0.0;

	try
	{
		fopen_s(&fp, strFilePath, "r");
		fscanf_s(fp, "row:%d,col:%d\n", &numRows, &numCols);
		matDistance = cv::Mat::zeros(numRows, numCols, CV_32FC1);

		for(unsigned int rowIdx = 0; rowIdx < (unsigned int)numRows; rowIdx++)
		{
			for(unsigned int colIdx = 0; colIdx < (unsigned int)numCols; colIdx++)
			{
				fscanf_s(fp, "%f,", &curSensitivity);
				matDistance.at<float>(rowIdx, colIdx) = curSensitivity;
			}
			fscanf_s(fp, "\n");
		}

		fclose(fp);
		return true;
	}
	catch(DWORD dwError)
	{
		printf("[ERROR] cannot open distance from boundary matrix of camera %d! error code %d\n", CAM_ID[camIdx], dwError);
	}
	return false;
}

/************************************************************************
 Method Name: CheckVisibility
 Description: 
	- return whether an input point can be seen at a specific camera
 Input Arguments:
	- testPoint: a 3D point
	- camIdx: camera index (for calibration information)
 Return Values:
	- bool: visible(true)/un-visible(false)
************************************************************************/
bool CPSNWhere_Associator3D::CheckVisibility(PSN_Point3D testPoint, unsigned int camIdx)
{
	PSN_Point2D reprojectedPoint = this->WorldToImage(testPoint, camIdx);
	if(reprojectedPoint.x < 0 || reprojectedPoint.x >= (double)cCamModel_[camIdx].width() || 
		reprojectedPoint.y < 0 || reprojectedPoint.y >= (double)cCamModel_[camIdx].height())
	{
		return false;
	}
	return true;
}

/************************************************************************
 Method Name: CheckHeadWidth
 Description: 
	- 
 Input Arguments:
	- 
	- 
 Return Values:
	- 
************************************************************************/
bool CPSNWhere_Associator3D::CheckHeadWidth(PSN_Point3D midPoint3D, PSN_Rect rect1, PSN_Rect rect2, unsigned int camIdx1, unsigned int camIdx2)
{
	PSN_Point2D ptCenter1 = rect1.center();
	PSN_Point2D ptCenter2 = rect2.center();
	PSN_Point2D ptL1(rect1.x, ptCenter1.y);
	PSN_Point2D ptL2(rect2.x, ptCenter2.y);
	PSN_Point2D ptR1(rect1.x + rect1.w, ptCenter1.y);
	PSN_Point2D ptR2(rect2.x + rect2.w, ptCenter2.y);
	PSN_Point3D ptL3D1 = this->ImageToWorld(ptL1, midPoint3D.z, camIdx1);
	PSN_Point3D ptR3D1 = this->ImageToWorld(ptR1, midPoint3D.z, camIdx1);
	PSN_Point3D ptL3D2 = this->ImageToWorld(ptL2, midPoint3D.z, camIdx2);
	PSN_Point3D ptR3D2 = this->ImageToWorld(ptR2, midPoint3D.z, camIdx2);
	double width3D1 = (ptL3D1 - ptR3D1).norm_L2();
	double width3D2 = (ptL3D2 - ptR3D2).norm_L2();

	if(MAX_HEAD_WIDTH < width3D1 && MAX_HEAD_WIDTH < width3D1)
	{
		return false;
	}
	if(MAX_HEAD_WIDTH < width3D1 && WIDTH_DIFF_RATIO_THRESHOLD * width3D1 > width3D2)
	{
		return false;
	}
	if(MAX_HEAD_WIDTH < width3D1 && WIDTH_DIFF_RATIO_THRESHOLD * width3D2 > width3D1)
	{
		return false;
	}
	return true;
}

/************************************************************************
 Method Name: PointReconstruction
 Description: 
	- Reconstruction 3D point with most recent points of 2D tracklets
 Input Arguments:
	- tracklet2Ds: 2D tracklets which are used to reconstruct 3D point
 Return Values:
	- stReconstruction: information structure of reconstructed point
************************************************************************/
stReconstruction CPSNWhere_Associator3D::PointReconstruction(CTrackletCombination &tracklet2Ds)
{
	stReconstruction resultReconstruction;
	resultReconstruction.bIsMeasurement = false;
	resultReconstruction.point = PSN_Point3D(0.0, 0.0, 0.0);
	resultReconstruction.costReconstruction = DBL_MAX;
	resultReconstruction.costLink = 0.0;
	resultReconstruction.velocity = PSN_Point3D(0.0, 0.0, 0.0);
	
	if(0 == tracklet2Ds.size())
	{
		return resultReconstruction;
	}

	resultReconstruction.bIsMeasurement = true;
	resultReconstruction.tracklet2Ds = tracklet2Ds;	
	double fDistance = DBL_MAX;
	double maxError = E_CAL;
	double probabilityReconstruction = 0.0;
	PSN_Point2D curPoint(0.0, 0.0);	

	//-------------------------------------------------
	// POINT RECONSTRUCTION
	//-------------------------------------------------
	switch(PSN_DETECTION_TYPE)
	{
	case 1:
		// Full-body
		{						
			std::vector<PSN_Point2D_CamIdx> vecPointInfos;
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				stTracklet2D *curTracklet = resultReconstruction.tracklet2Ds.get(camIdx);
				if(NULL == curTracklet){ continue; }

				curPoint = curTracklet->rects.back().reconstructionPoint();			
				vecPointInfos.push_back(PSN_Point2D_CamIdx(curPoint, camIdx));
				//maxError += E_DET * matProjectionSensitivity_[camIdx].at<float>((int)curPoint.y, (int)curPoint.x);
			}
			maxError = MAX_TRACKLET_SENSITIVITY_ERROR;
			fDistance = this->NViewGroundingPointReconstruction(vecPointInfos, resultReconstruction.point);
		}		
		break;
	default:
		// Head
		{
			std::vector<PSN_Line> vecBackprojectionLines;
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				stTracklet2D *curTracklet = resultReconstruction.tracklet2Ds.get(camIdx);
				if(NULL == curTracklet){ continue; }
								
				curPoint = curTracklet->rects.back().reconstructionPoint(); 
				PSN_Line curLine = curTracklet->backprojectionLines.back();

				vecBackprojectionLines.push_back(curLine);
				maxError += E_DET * matProjectionSensitivity_[camIdx].at<float>((int)curPoint.y, (int)curPoint.x);
			}
			fDistance = this->NViewPointReconstruction(vecBackprojectionLines, resultReconstruction.point);
		}
		break;
	}

	if(2 > tracklet2Ds.size())
	{
		probabilityReconstruction = 0.5;
	}
	else
	{
		if(fDistance > maxError)
		{
			return resultReconstruction;
		}
		probabilityReconstruction = 1 == resultReconstruction.tracklet2Ds.size()? 0.5 : 0.5 * psn::erfc(4.0 * fDistance / maxError - 2.0);
	}

	//-------------------------------------------------
	// DETECTION PROBABILITY
	//-------------------------------------------------
	double fDetectionProbabilityRatio = 1.0;
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(!this->CheckVisibility(resultReconstruction.point, camIdx))
		{
			continue;
		}

		if(NULL == resultReconstruction.tracklet2Ds.get(camIdx))
		{
			// false negative
			fDetectionProbabilityRatio *= FN_RATE / (1 - FN_RATE);
			continue;
		}
		// positive
		fDetectionProbabilityRatio *= (1 - FP_RATE) / FP_RATE;
	}

	// reconstruction cost
	resultReconstruction.costReconstruction = log(1 - probabilityReconstruction) - log(probabilityReconstruction) - log(fDetectionProbabilityRatio);

	return resultReconstruction;
}

/************************************************************************
 Method Name: NViewPointReconstruction
 Description: 
	- reconstruct 3D point with multiple back projection lines
 Input Arguments:
	- vecLines: vector of back projeciton lines
	- outputPoint: (output) reconstructed point
 Return Values:
	- double: return distance between back projection lines and reconstructed point
************************************************************************/
double CPSNWhere_Associator3D::NViewPointReconstruction(std::vector<PSN_Line> &vecLines, PSN_Point3D &outputPoint)
{	
#ifdef PSN_DEBUG_MODE_
	//printf("[CPSNWhere_Associator3D](MultiViewPointReconstruction) start\n");
#endif

	unsigned int numLines = (unsigned int)vecLines.size();

	if(numLines < 2)
	{
		outputPoint = vecLines[0].second;
		return MAX_TRACKLET_DISTANCE/2.0;
	}


	cv::Mat P = cv::Mat::zeros(3, 3, CV_64FC1);
	cv::Mat PP = cv::Mat::zeros(3, 3, CV_64FC1);
	cv::Mat A = cv::Mat::zeros(3, 3, CV_64FC1);	
	cv::Mat b = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat resultPoint = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat I3x3 = cv::Mat::eye(3, 3, CV_64FC1);
	std::deque<cv::Mat> vecV;
	
	// reconstruction
	PSN_Point3D vecDir;
	for(unsigned int lineIdx = 0; lineIdx < numLines; lineIdx++)
	{
		vecDir = vecLines[lineIdx].second - vecLines[lineIdx].first;
		vecDir /= vecDir.norm_L2();
		cv::Mat v = cv::Mat(vecDir.cv());
		vecV.push_back(v);
		cv::Mat s = cv::Mat(vecLines[lineIdx].first.cv());
		P = (v * v.t() - I3x3);
		PP = P.t() * P;
		A = A + PP;
		b = b + PP * s;
	}
	resultPoint = A.inv() * b;
	outputPoint.x = resultPoint.at<double>(0, 0);
	outputPoint.y = resultPoint.at<double>(1, 0);
	outputPoint.z = resultPoint.at<double>(2, 0);

	// error measurement
	double resultError = 0.0;
	double lambda = 0;
	cv::Mat diffVec = cv::Mat(3, 1, CV_64FC1);
	PSN_Point3D diffPoint(0, 0, 0);
	for(unsigned int lineIdx = 0; lineIdx < numLines; lineIdx++)
	{
		cv::Mat s = cv::Mat(vecLines[lineIdx].first.cv());
		lambda = vecV[lineIdx].dot(resultPoint - s);
		diffVec = s + lambda * vecV[lineIdx] - resultPoint;
		diffPoint.x = diffVec.at<double>(0, 0);
		diffPoint.y = diffVec.at<double>(1, 0);
		diffPoint.z = diffVec.at<double>(2, 0);
		resultError += (1/(double)numLines)*diffPoint.norm_L2();
	}

#ifdef PSN_DEBUG_MODE_
	//printf("[CPSNWhere_Associator3D](MultiViewPointReconstruction) return (%f, %f, %f) with error %f\n", outputPoint.x, outputPoint.y, outputPoint.z, resultError);
#endif

	return resultError;
}


/************************************************************************
 Method Name: NViewGroundingPointReconstruction
 Description: 
	- reconstruct 3D point with 2D grounding points (for PETS 9002 dataset)
 Input Arguments:
	- vecBottomCenterPoints: vector of bottom center point of each object
	- outputPoint: (output) reconstructed point
 Return Values:
	- double: return distance between back projection lines and reconstructed point
************************************************************************/
double CPSNWhere_Associator3D::NViewGroundingPointReconstruction(std::vector<PSN_Point2D_CamIdx> &vecPointInfos, PSN_Point3D &outputPoint)
{
	double resultError = 0.0;
	double sumInvSensitivity = 0.0;
	unsigned int numPoints = (unsigned int)vecPointInfos.size();
	std::vector<PSN_Point3D> vecReconstructedPoints;
	std::vector<double> vecInvSensitivity;
	vecReconstructedPoints.reserve(numPoints);

	outputPoint = PSN_Point3D(0, 0, 0);
	for(unsigned int pointIdx = 0; pointIdx < numPoints; pointIdx++)
	{
		PSN_Point2D curPoint = vecPointInfos[pointIdx].first;
		unsigned int curCamIdx = vecPointInfos[pointIdx].second;

		int sensitivityX = std::min(std::max(0, (int)curPoint.x), matProjectionSensitivity_[curCamIdx].cols);
		int sensitivityY = std::min(std::max(0, (int)curPoint.y), matProjectionSensitivity_[curCamIdx].rows);
		double curSensitivity = matProjectionSensitivity_[curCamIdx].at<float>(sensitivityY, sensitivityX);
		double invSensitivity = 1.0 / curSensitivity;

		vecReconstructedPoints.push_back(this->ImageToWorld(curPoint, 0, curCamIdx));
		vecInvSensitivity.push_back(invSensitivity);
		outputPoint += vecReconstructedPoints.back() * invSensitivity;
		sumInvSensitivity += invSensitivity;
	}
	outputPoint /= sumInvSensitivity;
	
	if(2 > numPoints)
	{
		return MAX_TRACKLET_SENSITIVITY_ERROR;
	}

	for(unsigned int pointIdx = 0; pointIdx < numPoints; pointIdx++)
	{
		resultError += (1.0/(double)numPoints) * vecInvSensitivity[pointIdx] * (outputPoint - vecReconstructedPoints[pointIdx]).norm_L2();
	}

	//outputPoint = PSN_Point3D(0, 0, 0);
	//for(unsigned int pointIdx = 0; pointIdx < numPoints; pointIdx++)
	//{
	//	PSN_Point2D curPoint = vecPointInfos[pointIdx].first;
	//	unsigned int curCamIdx = vecPointInfos[pointIdx].second;

	//	vecReconstructedPoints.push_back(this->ImageToWorld(curPoint, 0, curCamIdx));
	//	outputPoint += vecReconstructedPoints.back() * (1.0/(double)numPoints);
	//}
	//
	//if(2 > numPoints)
	//{
	//	return MAX_TRACKLET_DISTANCE/2.0;
	//}

	//for(unsigned int pointIdx = 0; pointIdx < numPoints; pointIdx++)
	//{
	//	resultError = (1.0/(double)numPoints) * (outputPoint - vecReconstructedPoints[pointIdx]).norm_L2();
	//}

	return resultError;
}

/************************************************************************
 Method Name: GetBackProjectionLine
 Description: 
	- find two 3D points on the back-projection line correspondes to an input point
 Input Arguments:
	- point2D: image coordinate of the point
	- camIdx: camera index (for calibration information)
 Return Values:
	- PSL_Line: a pair of 3D points have a height 2000 and 0
************************************************************************/
PSN_Line CPSNWhere_Associator3D::GetBackProjectionLine(PSN_Point2D point2D, unsigned int camIdx)
{
	PSN_Point3D pointTop, pointBottom;
	pointTop = this->ImageToWorld(point2D, 2000, camIdx);
	pointBottom = this->ImageToWorld(point2D, 0, camIdx);	
	return PSN_Line(pointTop, pointBottom);
}

/************************************************************************
 Method Name: Tracklet2D_UpdateTracklets
 Description: 
	- Update tracklets in each camera with information from 2D tracker
 Input Arguments:
	- curTrack2DResult: vector of 2D tracking results
	- frameIdx: current frame index
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Tracklet2D_UpdateTracklets(std::vector<stTrack2DResult> &curTrack2DResult, unsigned int frameIdx)
{
#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](Update2DTracklets) start\n");
	time_t timer_start = clock();
#endif

	bReceiveNewMeasurement_ = false;
	nNumTotalActive2DTracklet_ = 0;

	// check the validity of tracking result
	if(NUM_CAM != curTrack2DResult.size()){ return;	}

	/////////////////////////////////////////////////////////////////////
	// 2D TRACKLET UPDATE
	/////////////////////////////////////////////////////////////////////
	// find update infos (real updating will be done after this loop) and generate new tracklets
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		//this->m_matDetectionMap[camIdx] = cv::Mat::zeros(this->m_matDetectionMap[camIdx].rows, this->m_matDetectionMap[camIdx].cols, CV_64FC1);
		vecTracklet2DSet_[camIdx].newMeasurements.clear();
		if(frameIdx != curTrack2DResult[camIdx].frameIdx){ continue; }
		if(camIdx != curTrack2DResult[camIdx].camID){ continue; }

		unsigned int numObject = (unsigned int)curTrack2DResult[camIdx].object2DInfos.size();
		unsigned int numTracklet = (unsigned int)vecTracklet2DSet_[camIdx].activeTracklets.size();
		std::vector<stObject2DInfo*> tracklet2DUpdateInfos(numTracklet, NULL);

		//-------------------------------------------------
		// MATCHING AND GENERATING NEW 2D TRACKLET
		//-------------------------------------------------
		for(unsigned int objectIdx = 0; objectIdx < numObject; objectIdx++)
		{
			stObject2DInfo *curObject = &curTrack2DResult[camIdx].object2DInfos[objectIdx];

			// detection size crop
			curObject->box = curObject->box.cropWithSize(cCamModel_[camIdx].width(), cCamModel_[camIdx].height());

			// set detection map
			//this->m_matDetectionMap[camIdx](curObject->box.cv()) = 1.0;

			// find appropriate 2D tracklet
			bool bNewTracklet2D = true; 
			for(unsigned int tracklet2DIdx = 0; tracklet2DIdx < numTracklet; tracklet2DIdx++)
			{
				if(curObject->id == vecTracklet2DSet_[camIdx].activeTracklets[tracklet2DIdx]->id)
				{
					bNewTracklet2D = false;
					tracklet2DUpdateInfos[tracklet2DIdx] = curObject;
					break;
				}
			}
			if(!bNewTracklet2D){ continue; }

			// generate a new 2D tracklet
			stTracklet2D newTracklet;			
			newTracklet.id = curObject->id;
			newTracklet.camIdx = camIdx;
			newTracklet.bActivated = true;
			newTracklet.rects.push_back(curObject->box);
			newTracklet.backprojectionLines.push_back(this->GetBackProjectionLine(curObject->box.center(), camIdx));
			newTracklet.timeStart = frameIdx;
			newTracklet.timeEnd = frameIdx;			
			newTracklet.duration = 1;

			// update tracklet info
			vecTracklet2DSet_[camIdx].tracklets.push_back(newTracklet);
			vecTracklet2DSet_[camIdx].activeTracklets.push_back(&vecTracklet2DSet_[camIdx].tracklets.back());			
			vecTracklet2DSet_[camIdx].newMeasurements.push_back(&vecTracklet2DSet_[camIdx].tracklets.back());
			nNumTotalActive2DTracklet_++;			
			bReceiveNewMeasurement_ = true;
		}

		//-------------------------------------------------
		// UPDATING ESTABLISHED 2D TRACKLETS
		//-------------------------------------------------
		std::deque<stTracklet2D*>::iterator trackletIter = vecTracklet2DSet_[camIdx].activeTracklets.begin();
		for(unsigned int objInfoIdx = 0; objInfoIdx < numTracklet; objInfoIdx++)
		{
			if(NULL == tracklet2DUpdateInfos[objInfoIdx])
			{
				if(!(*trackletIter)->bActivated)
				{					
					trackletIter = vecTracklet2DSet_[camIdx].activeTracklets.erase(trackletIter);				
					continue;
				}			
				(*trackletIter)->bActivated = false;
				trackletIter++;
				continue;
			}

			// update tracklet
			stTracklet2D *curTracklet = *trackletIter;
			stObject2DInfo *curObject = tracklet2DUpdateInfos[objInfoIdx];
			curTracklet->bActivated = true;
			curTracklet->rects.push_back(curObject->box);
			curTracklet->backprojectionLines.push_back(this->GetBackProjectionLine(curObject->box.center(), camIdx));
			curTracklet->timeEnd = frameIdx;
			curTracklet->duration++;

			// association informations
			for(unsigned int subloopCamIdx = 0; subloopCamIdx < NUM_CAM; subloopCamIdx++)
			{
				curTracklet->bAssociableNewMeasurement[subloopCamIdx].clear();
			}

			// increase iterators
			trackletIter++;
		}
	}

	/////////////////////////////////////////////////////////////////////
	// ASSOCIATION CHECK
	/////////////////////////////////////////////////////////////////////
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(!bReceiveNewMeasurement_){ break; }

		for(std::deque<stTracklet2D*>::iterator trackletIter = vecTracklet2DSet_[camIdx].activeTracklets.begin();
			trackletIter != vecTracklet2DSet_[camIdx].activeTracklets.end();
			trackletIter++)
		{		
			PSN_Line backProjectionLine1 = (*trackletIter)->backprojectionLines.back();
			for(unsigned int subloopCamIdx = 0; subloopCamIdx < NUM_CAM; subloopCamIdx++)
			{
				unsigned int numNewMeasurements = (unsigned int)vecTracklet2DSet_[subloopCamIdx].newMeasurements.size();
				(*trackletIter)->bAssociableNewMeasurement[subloopCamIdx].resize(numNewMeasurements, false);

				if(camIdx == subloopCamIdx){ continue; }
				for(unsigned int measurementIdx = 0; measurementIdx < numNewMeasurements; measurementIdx++)
				{
					stTracklet2D *curMeasurement = vecTracklet2DSet_[subloopCamIdx].newMeasurements[measurementIdx];
					PSN_Line backProjectionLine2 = curMeasurement->backprojectionLines.back();

					PSN_Point3D reconstructedPoint;
					std::vector<PSN_Line> vecBackProjectionLines;
					vecBackProjectionLines.push_back(backProjectionLine1);
					vecBackProjectionLines.push_back(backProjectionLine2);

					double fDistance = this->NViewPointReconstruction(vecBackProjectionLines, reconstructedPoint);

					//double fDistance = this->StereoTrackletReconstruction(*trackletIter, curMeasurement, reconstructedPoints);
					

					if(MAX_TRACKLET_DISTANCE >= fDistance)
					{
						(*trackletIter)->bAssociableNewMeasurement[subloopCamIdx][measurementIdx] = true;
					}
				}
			}
		}
	}

#ifdef PSN_DEBUG_MODE_
	time_t timer_end = clock();
	double processingTime = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
	printf("[CPSNWhere_Associator3D](Update2DTracklets) processing time: %f\n", processingTime);
#endif
}

/************************************************************************
 Method Name: GenerateTrackletCombinations
 Description: 
	- Generate feasible 2D tracklet combinations
 Input Arguments:
	- vecBAssociationMap: feasible association map
	- combination: current combination
	- combinationQueue: queue for save combinations
	- camIdx: current camera index
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::GenerateTrackletCombinations(std::vector<bool> *vecBAssociationMap, 
														  CTrackletCombination combination, 
														  std::deque<CTrackletCombination> &combinationQueue, 
														  unsigned int camIdx)
{
	if(camIdx >= NUM_CAM)
	{
#ifdef PSN_DEBUG_MODE
		//combination.print();
#endif
		combinationQueue.push_back(combination);
		return;
	}

	if(NULL != combination.get(camIdx))
	{
		std::vector<bool> vecNewBAssociationMap[NUM_CAM];
		for(unsigned int subloopCamIdx = 0; subloopCamIdx < NUM_CAM; subloopCamIdx++)
		{
			vecNewBAssociationMap[subloopCamIdx] = vecBAssociationMap[subloopCamIdx];
			if(subloopCamIdx <= camIdx){ continue; }
			for(unsigned int mapIdx = 0; mapIdx < vecNewBAssociationMap[subloopCamIdx].size(); mapIdx++)
			{
				bool currentFlag = vecNewBAssociationMap[subloopCamIdx][mapIdx];
				bool measurementFlag = combination.get(camIdx)->bAssociableNewMeasurement[subloopCamIdx][mapIdx];
				vecNewBAssociationMap[subloopCamIdx][mapIdx] = currentFlag & measurementFlag;
			}
		}
		this->GenerateTrackletCombinations(vecNewBAssociationMap, combination, combinationQueue, camIdx+1);
		return;
	}

	// null tracklet
	combination.set(camIdx, NULL);
	this->GenerateTrackletCombinations(vecBAssociationMap, combination, combinationQueue, camIdx+1);

	for(unsigned int measurementIdx = 0; measurementIdx < vecTracklet2DSet_[camIdx].newMeasurements.size(); measurementIdx++)
	{
		if(!vecBAssociationMap[camIdx][measurementIdx]){ continue; }

		combination.set(camIdx, vecTracklet2DSet_[camIdx].newMeasurements[measurementIdx]);
		// AND operation of map
		std::vector<bool> vecNewBAssociationMap[NUM_CAM];
		for(unsigned int subloopCamIdx = 0; subloopCamIdx < NUM_CAM; subloopCamIdx++)
		{
			vecNewBAssociationMap[subloopCamIdx] = vecBAssociationMap[subloopCamIdx];
			if(subloopCamIdx <= camIdx){ continue; }
			for(unsigned int mapIdx = 0; mapIdx < vecNewBAssociationMap[subloopCamIdx].size(); mapIdx++)
			{
				bool currentFlag = vecNewBAssociationMap[subloopCamIdx][mapIdx];
				bool measurementFlag = vecTracklet2DSet_[camIdx].newMeasurements[measurementIdx]->bAssociableNewMeasurement[subloopCamIdx][mapIdx];
				vecNewBAssociationMap[subloopCamIdx][mapIdx] = currentFlag & measurementFlag;
			}
		}
		this->GenerateTrackletCombinations(vecNewBAssociationMap, combination, combinationQueue, camIdx+1);
	}
}

/************************************************************************
 Method Name: Track3D_Management
 Description:
	- updage established 3D tracks
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Track3D_Management(PSN_TrackSet &outputSeedTracks)
{
	outputSeedTracks.clear();

	// update 3D tracks
	this->Track3D_UpdateTracks();

	// generate 3D tracks
	if(bReceiveNewMeasurement_)
	{
		this->Track3D_GenerateSeedTracks(outputSeedTracks);
		this->Track3D_BranchTracks(&outputSeedTracks);
	}

	// track print
	char strTrackFileName[128];
	sprintf_s(strTrackFileName, "logs/tracks/%04d.txt", nCurrentFrameIdx_);
	PrintTracks(queueTracksInWindow_, strTrackFileName, false);
}

/************************************************************************
 Method Name: Track3D_UpdateTracks
 Description:
	- updage established 3D tracks
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Track3D_UpdateTracks(void)
{
#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](UpdateTracks) start\n");
	time_t timer_start = clock();
#endif
	std::deque<Track3D*> queueNewTracks;

	//---------------------------------------------------------
	// UPDATE ACTIVE TRACKS
	//---------------------------------------------------------	
	for(std::deque<Track3D*>::iterator trackIter = queueActiveTrack_.begin();
		trackIter != queueActiveTrack_.end();
		trackIter++)
	{
		if(!(*trackIter)->bValid)
		{
			continue;
		}
		Track3D *curTrack = *trackIter;

		// updating current tracklet information
		for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if(NULL == curTrack->curTracklet2Ds.get(camIdx)){ continue; }
			if(!curTrack->curTracklet2Ds.get(camIdx)->bActivated)
			{
				if(MIN_TRACKLET_LENGTH > curTrack->curTracklet2Ds.get(camIdx)->duration)
				{
					// invalidate track with 2D tracklet which has duration 1
					TrackTree::SetValidityFlagInTrackBranch(curTrack, false);
					break;
				}
				curTrack->curTracklet2Ds.set(camIdx, NULL);
			}
		}
		if(!curTrack->bValid)
		{			
			continue;
		}	

		// activation check and expiration
		if(0 == curTrack->curTracklet2Ds.size())
		{ 		
			// de-activating
			curTrack->bActive = false;

			// cost update (with exit cost)
			std::vector<PSN_Point2D_CamIdx> vecLastPointInfo;
			stReconstruction lastReconstruction = curTrack->reconstructions.back();
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				if(NULL == lastReconstruction.tracklet2Ds.get(camIdx))
				{
					continue;
				}
				vecLastPointInfo.push_back(PSN_Point2D_CamIdx(lastReconstruction.tracklet2Ds.get(camIdx)->rects.back().center(), camIdx));
			}			
			curTrack->costExit = std::min(COST_EX_MAX, -log(this->ComputeExitProbability(vecLastPointInfo)));
			curTrack->costTotal = curTrack->costEnter 
								+ curTrack->costReconstruction 
								+ curTrack->costLink 
								+ curTrack->costExit;

			// move to time jump track list
			queuePausedTrack_.push_back(curTrack);
			continue;
		}
		
		// time related information
		curTrack->duration++;
		curTrack->timeEnd++;

		// point reconstruction and cost update
		stReconstruction curReconstruction = this->PointReconstruction(curTrack->curTracklet2Ds);

		//double curLinkProbability = ComputeLinkProbability(curTrack->reconstructions.back().point + curTrack->reconstructions.back().velocity, curReconstruction.point, 1);
		double curLinkProbability = ComputeLinkProbability(curTrack->reconstructions.back().point, curReconstruction.point, 1);
		if(DBL_MAX == curReconstruction.costReconstruction || MIN_LINKING_PROBABILITY > curLinkProbability)
		{
			// invalidate track			
			curTrack->bValid = false;			
			//this->SetValidityFlagInTrackBranch(curTrack, false);
			continue;
		}
		curReconstruction.costLink = -log(curLinkProbability);
		curTrack->costReconstruction += curReconstruction.costReconstruction;
		curTrack->costLink += curReconstruction.costLink;

		// velocity update
		curReconstruction.velocity = curTrack->reconstructions.back().velocity * (1 - VELOCITY_LEARNING_RATE) + (curReconstruction.point - curTrack->reconstructions.back().point) * VELOCITY_LEARNING_RATE;

		//// update Kalman Filter
		//PSN_Point3D curPoint = curReconstruction.point;	
		//cv::Mat curPrediction = curTrack->KF.predict();
		//curTrack->KFMeasurement.at<float>(0, 0) = (float)curPoint.x;
		//curTrack->KFMeasurement.at<float>(1, 0) = (float)curPoint.y;
		//curTrack->KFMeasurement.at<float>(2, 0) = (float)curPoint.z;
		//curTrack->KF.correct(curTrack->KFMeasurement);

		//// refinement with Kalman filter estimation
		//curPoint.x = (double)curTrack->KF.statePost.at<float>(0, 0);
		//curPoint.y = (double)curTrack->KF.statePost.at<float>(1, 0);
		//curPoint.z = (double)curTrack->KF.statePost.at<float>(2, 0);
		//curReconstruction.point = curPoint;
		curTrack->reconstructions.push_back(curReconstruction);

		// increase iterator
		queueNewTracks.push_back(curTrack);
	}
	queueActiveTrack_ = queueNewTracks;
	queueNewTracks.clear();


	//---------------------------------------------------------
	// UPDATE DE-ACTIVATED TRACKS FOR TEMPORAL BRANCHING
	//---------------------------------------------------------
	std::deque<Track3D*> queueTerminatedTracks;
	for(std::deque<Track3D*>::iterator trackIter = queuePausedTrack_.begin();
		trackIter != queuePausedTrack_.end();
		trackIter++)
	{
		if(!(*trackIter)->bValid)
		{
			continue;
		}

		// handling expired track
		if((*trackIter)->timeEnd + MAX_TIME_JUMP < nCurrentFrameIdx_)
		{
			// remove track instance for memory efficiency
			if(0.0 <= (*trackIter)->costTotal)
			{
				(*trackIter)->bValid = false;
			}
			else
			{
				// for logging
				queueTerminatedTracks.push_back(*trackIter);
			}
			continue;
		}

		//// update Kalman Filter
		//cv::Mat curPrediction = (*trackIter)->KF.predict();
		//PSN_Point3D curPredictedPoint(
		//	(double)curPrediction.at<float>(0, 0), 
		//	(double)curPrediction.at<float>(1, 0), 
		//	(double)curPrediction.at<float>(2, 0));
		////(*trackIter)->KFPrediction.push_back(curPredictedPoint);

		// insert dummy reconstruction with Kalman prediction
		stReconstruction dummyReconstruction = this->PointReconstruction((*trackIter)->curTracklet2Ds);
		dummyReconstruction.bIsMeasurement = false;
		//dummyReconstruction.point = curPredictedPoint;
		dummyReconstruction.point = (*trackIter)->reconstructions.back().point;
		(*trackIter)->reconstructions.push_back(dummyReconstruction);
						
		queueNewTracks.push_back(*trackIter);
	}
	queuePausedTrack_ = queueNewTracks;
	queueNewTracks.clear();

	// print out
#ifdef PSN_PRINT_LOG
	//this->PrintTracks(queueTerminatedTracks, strTrackLogFileName_, true);
#endif
	queueTerminatedTracks.clear();

	//---------------------------------------------------------
	// MANAGE TRACKS IN PROCESSING WINDOW
	//---------------------------------------------------------
	for(std::deque<Track3D*>::iterator trackIter = queueTracksInWindow_.begin();
		trackIter != queueTracksInWindow_.end();
		trackIter++)
	{
		if(!(*trackIter)->bValid
			|| (*trackIter)->timeEnd + PROC_WINDOW_SIZE <= nCurrentFrameIdx_)
		{
			continue;
		}
		queueNewTracks.push_back(*trackIter);
		if((*trackIter)->tree->maxGTProb < (*trackIter)->GTProb)
		{
			(*trackIter)->tree->maxGTProb = (*trackIter)->GTProb;
		}
	}
	queueTracksInWindow_ = queueNewTracks;
	queueNewTracks.clear();
	
	//---------------------------------------------------------
	// UPDATE TRACK TREES
	//---------------------------------------------------------
	// delete empty trees from the active tree list
	std::deque<TrackTree*> queueNewActiveTrees;
	for (std::deque<TrackTree*>::iterator treeIter = queuePtActiveTrees_.begin();
		treeIter != queuePtActiveTrees_.end();
		treeIter++)
	{
		// track validation
		std::deque<Track3D*> queueUpdated; // copy is faster than delete
		for (std::deque<Track3D*>::iterator trackIter = (*treeIter)->tracks.begin();
			trackIter != (*treeIter)->tracks.end();
			trackIter++)
		{
			// reset fields for optimization
			(*trackIter)->BranchGTProb = 0.0;
			(*trackIter)->GTProb = 0.0;
			(*trackIter)->bCurrentBestSolution = false;

			if (!(*trackIter)->bValid){ continue; }
			queueUpdated.push_back(*trackIter);
		}

		// update 2D tracklet info
		(*treeIter)->numMeasurements = 0;
		for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if (0 == (*treeIter)->tracklet2Ds[camIdx].size()) { continue; }

			std::deque<stTracklet2DInfo> queueUpdatedTrackletInfo;
			for(std::deque<stTracklet2DInfo>::iterator infoIter = (*treeIter)->tracklet2Ds[camIdx].begin();
				infoIter != (*treeIter)->tracklet2Ds[camIdx].end();
				infoIter++)
			{
				if((*infoIter).tracklet2D->timeStart + nNumFramesForProc_ < nCurrentFrameIdx_)
				{
					continue;
				}

				std::deque<Track3D*> queueNewTracks;
				for(std::deque<Track3D*>::iterator trackIter = (*infoIter).queueRelatedTracks.begin();
					trackIter != (*infoIter).queueRelatedTracks.end();
					trackIter++)
				{
					if((*trackIter)->bValid)
					{
						queueNewTracks.push_back(*trackIter);
					}
				}

				if(0 == queueNewTracks.size()){ continue; }

				(*infoIter).queueRelatedTracks = queueNewTracks;
				queueUpdatedTrackletInfo.push_back(*infoIter);
			}
			(*treeIter)->numMeasurements += (unsigned int)queueUpdatedTrackletInfo.size();
			(*treeIter)->tracklet2Ds[camIdx] = queueUpdatedTrackletInfo;
		}

		if(0 == queueUpdated.size())
		{
			(*treeIter)->bValid = false;
			continue;
		}

		(*treeIter)->tracks = queueUpdated;
		queueNewActiveTrees.push_back(*treeIter);
	}	
	queuePtActiveTrees_ = queueNewActiveTrees;
	queueNewActiveTrees.clear();

	// update unconfirmed trees
	std::deque<TrackTree*> queueNewUnconfirmedTrees;
	for(std::deque<TrackTree*>::iterator treeIter = queuePtUnconfirmedTrees_.begin();
		treeIter != queuePtUnconfirmedTrees_.end();
		treeIter++)
	{
		if(!(*treeIter)->bValid || (*treeIter)->timeGeneration + NUM_FRAME_FOR_CONFIRMATION <= nCurrentFrameIdx_)
		{ continue; }
		queueNewUnconfirmedTrees.push_back(*treeIter);
	}
	queuePtUnconfirmedTrees_ = queueNewUnconfirmedTrees;
	queueNewUnconfirmedTrees.clear();

	//---------------------------------------------------------
	// UPDATE HYPOTHESES
	//---------------------------------------------------------
	PSN_HypothesisSet validHypothesesSet;
	for (PSN_HypothesisSet::iterator hypothesisIter = queuePrevGlobalHypotheses_.begin();
		hypothesisIter != queuePrevGlobalHypotheses_.end();
		hypothesisIter++)
	{
		// validation
		for (size_t trackIdx = 0; trackIdx < (*hypothesisIter).selectedTracks.size(); trackIdx++)
		{
			if ((*hypothesisIter).selectedTracks[trackIdx]->bValid) { continue; }
			(*hypothesisIter).bValid = false;
			break;
		}
		if (!(*hypothesisIter).bValid) { continue; }

		// update related tracks
		PSN_TrackSet newRelatedTrackSet;
		for (size_t trackIdx = 0; trackIdx < (*hypothesisIter).relatedTracks.size(); trackIdx++)
		{
			if (!(*hypothesisIter).relatedTracks[trackIdx]->bValid) { continue; }
			newRelatedTrackSet.push_back((*hypothesisIter).relatedTracks[trackIdx]);
		}
		(*hypothesisIter).relatedTracks = newRelatedTrackSet;
		validHypothesesSet.push_back(*hypothesisIter);
	}
	queuePrevGlobalHypotheses_ = validHypothesesSet;

	//---------------------------------------------------------
	// UPDATE TRACK LIST
	//---------------------------------------------------------
	// delete invalid tracks (in the list of track instances)
	for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
		trackIter != listTrack3D_.end();
		/* do in the loop */)
	{		
		// delete invalid instance
		if (!trackIter->bValid)
		{			
			trackIter = listTrack3D_.erase(trackIter);
			continue;
		}

		// clear new flag
		trackIter->bNewTrack = false;
		// update the list of children tracks
		trackIter->childrenTrack = Track3D::GatherValidChildrenTracks(&(*trackIter), trackIter->childrenTrack);
		trackIter++;		
	}

#ifdef PSN_DEBUG_MODE_
	time_t timer_end = clock();
	double processingTime = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
	printf("[CPSNWhere_Associator3D](UpdateTracks) activeTrack: %d, time: %fs\n", queueActiveTrack_.size(), processingTime);
#endif
}


/************************************************************************
 Method Name: Track3D_GenerateSeedTracks
 Description: 
	- Generate seeds of track tree with combinations of new measurements
 Input Arguments:
	- outputSeedTracks: (output) generated seed tracks
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Track3D_GenerateSeedTracks(PSN_TrackSet &outputSeedTracks)
{
#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](GenerateSeeds) start\n");
	time_t timer_start = clock();
	unsigned int numPrevTracks = nNewTrackID_;
#endif

	// initialization
	outputSeedTracks.clear();

	//---------------------------------------------------------
	// FEASIBLE COMBINATIONS
	//---------------------------------------------------------
	std::deque<CTrackletCombination> queueSeeds;

	CTrackletCombination curCombination = CTrackletCombination();
	std::vector<bool> vecNullAssociateMap[NUM_CAM];

	// null tracklet at the first camera
	vecNullAssociateMap[0].resize(vecTracklet2DSet_[0].newMeasurements.size(), false);
	for(unsigned int camIdx = 1; camIdx < NUM_CAM; camIdx++)
	{
		vecNullAssociateMap[camIdx].resize(vecTracklet2DSet_[camIdx].newMeasurements.size(), true);
	}
	this->GenerateTrackletCombinations(vecNullAssociateMap, curCombination, queueSeeds, 1);

	// real tracklets at the first camera
	for(std::deque<stTracklet2D*>::iterator trackletIter = vecTracklet2DSet_[0].newMeasurements.begin();
		trackletIter != vecTracklet2DSet_[0].newMeasurements.end();
		trackletIter++)
	{
		curCombination.set(0, *trackletIter);
		this->GenerateTrackletCombinations((*trackletIter)->bAssociableNewMeasurement, curCombination, queueSeeds, 1);
	}

	// delete null track
	queueSeeds.erase(queueSeeds.begin());

	//---------------------------------------------------------
	// MAKE TRACKS AND TREES
	//---------------------------------------------------------
	for(unsigned int seedIdx = 0; seedIdx < queueSeeds.size(); seedIdx++)
	{
		// generate track
		Track3D newTrack;
		newTrack.Initialize(nNewTrackID_, NULL, nCurrentFrameIdx_, queueSeeds[seedIdx]);

		// tracklet information
		std::vector<PSN_Point2D_CamIdx> vecStartPointInfo;
		for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if(NULL == newTrack.curTracklet2Ds.get(camIdx)){ continue; }
			newTrack.tracklet2DIDs[camIdx].push_back(newTrack.curTracklet2Ds.get(camIdx)->id);
			vecStartPointInfo.push_back(PSN_Point2D_CamIdx(newTrack.curTracklet2Ds.get(camIdx)->rects.front().center(), camIdx));
		}

		// initiation cost
		newTrack.costEnter = bInitiationPenaltyFree_? -log(P_EN_MAX) : std::min(COST_EN_MAX, -log(this->ComputeEnterProbability(vecStartPointInfo)));
		vecStartPointInfo.clear();

		// point reconstruction
		stReconstruction curReconstruction = this->PointReconstruction(newTrack.curTracklet2Ds);
		if(DBL_MAX == curReconstruction.costReconstruction)
		{
			continue;
		}
		newTrack.costReconstruction = curReconstruction.costReconstruction;		
		newTrack.reconstructions.push_back(curReconstruction);

		// Kalman filter
		//newTrack.SetKalmanFilter(curReconstruction.point);
		
		// generate a track instance
		listTrack3D_.push_back(newTrack);
		nNewTrackID_++;
		Track3D *curTrack = &listTrack3D_.back();

		// generate a new track tree
		TrackTree newTree;
		newTree.Initialize(nNewTreeID_++, curTrack, nCurrentFrameIdx_, listTrackTree_);		
		queuePtActiveTrees_.push_back(curTrack->tree);
		queuePtUnconfirmedTrees_.push_back(curTrack->tree);

		// insert to queues
		outputSeedTracks.push_back(curTrack);		
	}

#ifdef PSN_DEBUG_MODE_
	time_t timer_end = clock();
	double processingTime = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
	printf("[CPSNWhere_Associator3D](GenerateSeeds) newTrack: %d, time: %fs\n", outputSeedTracks.size(), processingTime);
#endif
}


/************************************************************************
 Method Name: Track3D_BranchTracks
 Description: 
	- Generate branches of track tree with combinations of new measurements
 Input Arguments:
	- seedTracks: seed tracks
	- outputBranchTracks: (output) track generated by branching step
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Track3D_BranchTracks(PSN_TrackSet *seedTracks)
{
#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](BranchTracks) start\n");
	time_t timer_start = clock();
#endif

	/////////////////////////////////////////////////////////////////////
	// SPATIAL BRANCHING
	/////////////////////////////////////////////////////////////////////
	PSN_TrackSet queueBranchTracks;
	std::sort(queueActiveTrack_.begin(), queueActiveTrack_.end(), psnTrackGTPandLLDescend);
	for(std::deque<Track3D*>::iterator trackIter = queueActiveTrack_.begin();
		trackIter != queueActiveTrack_.end();
		trackIter++)
	{		
		if(DO_BRANCH_CUT && MAX_TRACK_IN_OPTIMIZATION <= queueBranchTracks.size()) { break; }

		//---------------------------------------------------------
		// FIND SPATIAL ASSOCIATION
		//---------------------------------------------------------	
		CTrackletCombination curCombination = (*trackIter)->curTracklet2Ds;

		// association map
		std::vector<bool> vecAssociationMap[NUM_CAM];
		for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			vecAssociationMap[camIdx].resize(vecTracklet2DSet_[camIdx].newMeasurements.size(), true);
		}
		for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if(NULL == curCombination.get(camIdx)){ continue; }			
			for(unsigned int subloopCamIdx = 0; subloopCamIdx < NUM_CAM; subloopCamIdx++)
			{
				for(unsigned int flagIdx = 0; flagIdx < vecTracklet2DSet_[subloopCamIdx].newMeasurements.size(); flagIdx++)
				{
					bool curTrackFlag = curCombination.get(camIdx)->bAssociableNewMeasurement[subloopCamIdx][flagIdx];
					bool curFlag = vecAssociationMap[subloopCamIdx][flagIdx];
					vecAssociationMap[subloopCamIdx][flagIdx] = curFlag && curTrackFlag;
				}
			}
		}

		// find feasible branches
		std::deque<CTrackletCombination> queueBranches;
		this->GenerateTrackletCombinations(vecAssociationMap, curCombination, queueBranches, 0);

		if(0 == queueBranches.size()){ continue; }
		if(1 == queueBranches.size() && queueBranches[0] == curCombination){ continue; }

		//---------------------------------------------------------
		// BRANCHING
		//---------------------------------------------------------	
		for(std::deque<CTrackletCombination>::iterator branchIter = queueBranches.begin();
			branchIter != queueBranches.end();
			branchIter++)
		{
			if(*branchIter == curCombination){ continue; }

			// generate a new track with branching combination
			Track3D newTrack;
			newTrack.id = nNewTrackID_;
			newTrack.curTracklet2Ds = *branchIter;			

			// reconstruction
			double costReconstructionIncrease = 0.0;
			double costLinkIncrease = 0.0;

			stReconstruction curReconstruction = this->PointReconstruction(newTrack.curTracklet2Ds);
			if(DBL_MAX == curReconstruction.costReconstruction)
			{
				continue;
			}

			stReconstruction oldReconstruction = (*trackIter)->reconstructions.back();
			stReconstruction preReconstruction = (*trackIter)->reconstructions[(*trackIter)->reconstructions.size()-2];
			//double curLinkProbability = ComputeLinkProbability(preReconstruction.point + preReconstruction.velocity, curReconstruction.point, 1);
			double curLinkProbability = ComputeLinkProbability(preReconstruction.point, curReconstruction.point, 1);
			if(MIN_LINKING_PROBABILITY > curLinkProbability)
			{
				continue;
			}
			curReconstruction.costLink = -log(curLinkProbability);
			costReconstructionIncrease += curReconstruction.costReconstruction - oldReconstruction.costReconstruction;
			costLinkIncrease += curReconstruction.costLink - oldReconstruction.costLink;
			curReconstruction.velocity = preReconstruction.velocity * (1 - VELOCITY_LEARNING_RATE) + (curReconstruction.point - preReconstruction.point) * VELOCITY_LEARNING_RATE;

			newTrack.reconstructions = (*trackIter)->reconstructions;
			newTrack.reconstructions.pop_back();
			newTrack.reconstructions.push_back(curReconstruction);

			// cost
			newTrack.costTotal = 0.0;
			newTrack.costReconstruction = (*trackIter)->costReconstruction + costReconstructionIncrease;
			newTrack.costLink = (*trackIter)->costLink + costLinkIncrease;
			newTrack.costEnter = (*trackIter)->costEnter;
			newTrack.costExit = 0.0;

			// copy 2D tracklet history and proecssig for clustering
			std::deque<stTracklet2D*> queueNewlyInsertedTracklet2D;
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				newTrack.tracklet2DIDs[camIdx] = (*trackIter)->tracklet2DIDs[camIdx];
				if(NULL == newTrack.curTracklet2Ds.get(camIdx))
				{
					continue;
				}
				if(0 == newTrack.tracklet2DIDs[camIdx].size() || newTrack.tracklet2DIDs[camIdx].back() != newTrack.curTracklet2Ds.get(camIdx)->id)
				{
					newTrack.tracklet2DIDs[camIdx].push_back(newTrack.curTracklet2Ds.get(camIdx)->id);
					queueNewlyInsertedTracklet2D.push_back(newTrack.curTracklet2Ds.get(camIdx));
				}
			}

			newTrack.bActive = true;
			newTrack.bValid = true;
			newTrack.tree = (*trackIter)->tree;
			newTrack.parentTrack = *trackIter;
			newTrack.childrenTrack.clear();
			newTrack.timeStart = (*trackIter)->timeStart;
			newTrack.timeEnd = (*trackIter)->timeEnd;
			newTrack.timeGeneration = nCurrentFrameIdx_;
			newTrack.duration = (*trackIter)->duration;	
			newTrack.bWasBestSolution = true;
			newTrack.GTProb = (*trackIter)->GTProb;
			
			//// Kalman filter
			//newTrack.KF = (*trackIter)->KF;
			//newTrack.KFMeasurement = (*trackIter)->KFMeasurement;

			// generate track instance
			listTrack3D_.push_back(newTrack);
			nNewTrackID_++;
					
			// insert to the track tree and related lists
			Track3D *branchTrack = &listTrack3D_.back();
			(*trackIter)->tree->tracks.push_back(branchTrack);
			(*trackIter)->childrenTrack.push_back(branchTrack);
			queueTracksInWindow_.push_back(branchTrack);

			//// for clustering
			//if(0 < queueNewlyInsertedTracklet2D.size())
			//{
			//	for(std::deque<stTracklet2D*>::iterator trackletIter = queueNewlyInsertedTracklet2D.begin();
			//		trackletIter != queueNewlyInsertedTracklet2D.end();
			//		trackletIter++)
			//	{
			//		bool bFound = false;
			//		for(std::deque<stTracklet2DInfo>::reverse_iterator infoIter = branchTrack->tree->tracklet2Ds[(*trackletIter)->camIdx].rbegin();
			//			infoIter != branchTrack->tree->tracklet2Ds[(*trackletIter)->camIdx].rend();
			//			infoIter++)
			//		{
			//			if((*infoIter).tracklet2D == *trackletIter)
			//			{
			//				(*infoIter).queueRelatedTracks.push_back(branchTrack);
			//				bFound = true;
			//				break;
			//			}
			//		}
			//		if(bFound)
			//		{
			//			continue;
			//		}
			//		stTracklet2DInfo newTrackletInfo;
			//		newTrackletInfo.tracklet2D = *trackletIter;
			//		newTrackletInfo.queueRelatedTracks.push_back(branchTrack);
			//		branchTrack->tree->tracklet2Ds[(*trackletIter)->camIdx].push_back(newTrackletInfo);
			//		branchTrack->tree->numMeasurements++;
			//	}
			//}

			// insert to global queue
			queueBranchTracks.push_back(branchTrack);
		}
	}

	// insert branches to the active track list
	size_t numBranches = queueActiveTrack_.size();
	queueActiveTrack_.insert(queueActiveTrack_.end(), queueBranchTracks.begin(), queueBranchTracks.end());

	/////////////////////////////////////////////////////////////////////
	// TEMPORAL BRANCHING
	/////////////////////////////////////////////////////////////////////
	int numTemporalBranch = 0;
	std::sort(queuePausedTrack_.begin(), queuePausedTrack_.end(), psnTrackGTPandLLDescend);
	for(std::deque<Track3D*>::iterator trackIter = queuePausedTrack_.begin();
		trackIter != queuePausedTrack_.end();
		trackIter++)
	{
		if(DO_BRANCH_CUT && MAX_TRACK_IN_OPTIMIZATION <= numTemporalBranch) { break; }

		Track3D *curTrack = *trackIter;

		for(std::deque<Track3D*>::iterator seedTrackIter = seedTracks->begin();
			seedTrackIter != seedTracks->end();
			seedTrackIter++)
		{
			Track3D *seedTrack = *seedTrackIter;
			std::deque<stReconstruction> queueSeedReconstruction = (*seedTrackIter)->reconstructions;
			std::deque<stReconstruction> queueJointReconstructions;
			unsigned int lengthValidReconstructions = curTrack->duration;
			double costReconstructionIncrease = 0.0;
			double costLinkIncrease = 0.0;
			unsigned int seedTrackReconIdx = 0;


			////////////////////////////////////////////////////////////////////////
			unsigned int timeGap = seedTrack->timeStart - curTrack->timeEnd;
			stReconstruction lastMeasurementReconstruction = curTrack->reconstructions[curTrack->timeEnd - curTrack->timeStart];
			double curLinkProbability = ComputeLinkProbability(lastMeasurementReconstruction.point, seedTrack->reconstructions.front().point, timeGap);
			if(MIN_LINKING_PROBABILITY > curLinkProbability)
			{
				continue;
			}
			lengthValidReconstructions += timeGap - 1;

			// for linking prior
			if(timeGap > 1)
			{					
				double linkingPrior = 1.0;
				for(unsigned int reconIdx = curTrack->duration; reconIdx < lengthValidReconstructions; reconIdx++)
				{
					double probDetection = 1.0;
					for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
					{
						if(this->CheckVisibility(curTrack->reconstructions[reconIdx].point, camIdx))
						{
							probDetection *= FN_RATE;
						}
					}
					linkingPrior *= probDetection;
				}
				
				curLinkProbability *= linkingPrior;
				if(MIN_LINKING_PROBABILITY > curLinkProbability)
				{
					continue;
				}
			}
			queueSeedReconstruction[0].costLink = -log(curLinkProbability);
			costLinkIncrease = queueSeedReconstruction[0].costLink;		

			////////////////////////////////////////////////////////////////////////

			// generate a new track with branching combination
			Track3D newTrack;
			newTrack.id = nNewTrackID_;
			newTrack.curTracklet2Ds = 0 == queueJointReconstructions.size()? seedTrack->curTracklet2Ds : queueJointReconstructions.back().tracklet2Ds;
			newTrack.bActive = true;
			newTrack.bValid = true;
			newTrack.tree = curTrack->tree;
			newTrack.parentTrack = curTrack;
			newTrack.childrenTrack.clear();
			newTrack.timeStart = curTrack->timeStart;
			newTrack.timeEnd = seedTrack->timeEnd;
			newTrack.timeGeneration = nCurrentFrameIdx_;
			newTrack.duration = newTrack.timeEnd - newTrack.timeStart + 1;
			newTrack.bWasBestSolution = true;
			newTrack.GTProb = (*trackIter)->GTProb;

			// tracklet history
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				newTrack.tracklet2DIDs[camIdx] = curTrack->tracklet2DIDs[camIdx];
				if(NULL == seedTrack->curTracklet2Ds.get(camIdx))
				{
					continue;
				}
				if(0 == newTrack.tracklet2DIDs[camIdx].size() || newTrack.tracklet2DIDs[camIdx].back() != seedTrack->tracklet2DIDs[camIdx].back())
				{
					newTrack.tracklet2DIDs[camIdx].push_back(seedTrack->tracklet2DIDs[camIdx].back());
				}
			}

			// cost
			newTrack.costTotal = 0.0;
			newTrack.costReconstruction = curTrack->costReconstruction + seedTrack->costReconstruction + costReconstructionIncrease;
			newTrack.costLink = curTrack->costLink + seedTrack->costLink + costLinkIncrease;
			newTrack.costEnter = curTrack->costEnter;
			newTrack.costExit = 0.0;	

			// reconstruction
			newTrack.reconstructions.insert(
				newTrack.reconstructions.begin(), 
				curTrack->reconstructions.begin(), 
				curTrack->reconstructions.begin() + lengthValidReconstructions);
			newTrack.reconstructions.insert(
				newTrack.reconstructions.end(), 
				queueJointReconstructions.begin(), 
				queueJointReconstructions.end());
			newTrack.reconstructions.insert(
				newTrack.reconstructions.end(), 
				queueSeedReconstruction.begin() + seedTrackReconIdx, 
				queueSeedReconstruction.end());
		
			//// Kalman filter
			//newTrack.KF = curTrack->KF;
			//newTrack.KFMeasurement = curTrack->KFMeasurement;
			//unsigned int newReconIdx = lengthValidReconstructions;
			//for(unsigned int reconIdx = seedTrackReconIdx; reconIdx < queueSeedReconstruction.size(); reconIdx++)
			//{
			//	PSN_Point3D curPoint = queueSeedReconstruction[reconIdx].point;

			//	// update Kalman filter
			//	if(reconIdx != seedTrackReconIdx)
			//	{
			//		newTrack.KF.predict();
			//	}			
			//	newTrack.KFMeasurement.at<float>(0, 0) = (float)curPoint.x;
			//	newTrack.KFMeasurement.at<float>(1, 0) = (float)curPoint.y;
			//	newTrack.KFMeasurement.at<float>(2, 0) = (float)curPoint.z;
			//	newTrack.KF.correct(newTrack.KFMeasurement);

			//	//// refinement with Kalman estimation
			//	//curPoint.x = (double)newTrack.KF.statePost.at<float>(0, 0);
			//	//curPoint.y = (double)newTrack.KF.statePost.at<float>(1, 0);
			//	//curPoint.z = (double)newTrack.KF.statePost.at<float>(2, 0);
			//	//newTrack.reconstructions[newReconIdx].point = curPoint;
			//	newReconIdx++;
			//}

			// insert to the list of track instances
			listTrack3D_.push_back(newTrack);
			nNewTrackID_++;
					
			// insert to the track tree and related lists
			Track3D *branchTrack = &listTrack3D_.back();
			curTrack->tree->tracks.push_back(branchTrack);
			curTrack->childrenTrack.push_back(branchTrack);
			queueActiveTrack_.push_back(branchTrack);
			queueTracksInWindow_.push_back(branchTrack);
			numTemporalBranch++;

			//// for clustering
			//for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			//{
			//	if(NULL == branchTrack->curTracklet2Ds.get(camIdx))
			//	{
			//		continue;
			//	}

			//	bool bFound = false;
			//	for(std::deque<stTracklet2DInfo>::reverse_iterator infoIter = branchTrack->tree->tracklet2Ds[camIdx].rbegin();
			//		infoIter != branchTrack->tree->tracklet2Ds[camIdx].rend();
			//		infoIter++)
			//	{
			//		if((*infoIter).tracklet2D == branchTrack->curTracklet2Ds.get(camIdx))
			//		{
			//			(*infoIter).queueRelatedTracks.push_back(branchTrack);
			//			bFound = true;
			//			break;
			//		}
			//	}
			//	if(bFound)
			//	{
			//		continue;
			//	}
			//	stTracklet2DInfo newTrackletInfo;
			//	newTrackletInfo.tracklet2D = branchTrack->curTracklet2Ds.get(camIdx);
			//	newTrackletInfo.queueRelatedTracks.push_back(branchTrack);
			//	branchTrack->tree->tracklet2Ds[camIdx].push_back(newTrackletInfo);
			//	branchTrack->tree->numMeasurements++;
			//}
		}
	}

	// add seed tracks to active track list
	queueActiveTrack_.insert(queueActiveTrack_.end(), seedTracks->begin(), seedTracks->end());
	queueTracksInWindow_.insert(queueTracksInWindow_.end(), seedTracks->begin(), seedTracks->end());

#ifdef PSN_DEBUG_MODE_
	time_t timer_end = clock();
	double processingTime = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
	numBranches = queueActiveTrack_.size() - numBranches;
	printf("[CPSNWhere_Associator3D](BranchTracks) new branches: %d, time: %fs\n", numBranches, processingTime);
#endif
}

/************************************************************************
 Method Name: Track3D_GetWholeCandidateTracks
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- PSN_TrackSet: selected tracks
************************************************************************/
PSN_TrackSet CPSNWhere_Associator3D::Track3D_GetWholeCandidateTracks(void)
{
	return queueTracksInWindow_;
}

///************************************************************************
// Method Name: Track3D_SolveHOMHT
// Description: 
//	- solve multiple hyphothesis problem with track trees
// Input Arguments:
//	- none
// Return Values:
//	- none
//************************************************************************/
//void CPSNWhere_Associator3D::Track3D_SolveHOMHT(void)
//{
//	if (0 == queueTracksInWindow_.size()) { return; }
//	
//	//---------------------------------------------------------
//	// PROPAGATE GLOBAL HYPOTHESES
//	//---------------------------------------------------------
//	if (0 == queuePrevGlobalHypotheses_.size())
//	{
//		this->Hypothesis_BranchHypotheses(this->Track3D_GetWholeCandidateTracks());
//	}
//	else
//	{	
//		// solve K times
//#pragma omp parallel for
//		for (std::deque<stGlobalHypothesis>::iterator hypothesisIter = queuePrevGlobalHypotheses_.begin();
//			hypothesisIter != queuePrevGlobalHypotheses_.end();
//			hypothesisIter++)
//		{
//			this->Hypothesis_BranchHypotheses((*hypothesisIter).relatedTracks, &((*hypothesisIter).selectedTracks));
//		}
//	}
//
//	if (0 == queueStGraphSolutions_.size()) { return; }
//
//	// sort solutions by probability and save best solution
//	std::sort(queueStGraphSolutions_.begin(), queueStGraphSolutions_.end(), psnSolutionLogLikelihoodDescendComparator);
//	if (0 < queueStGraphSolutions_.size())
//	{
//		for (PSN_TrackSet::iterator trackIter = queueStGraphSolutions_.front().tracks.begin();
//			trackIter != queueStGraphSolutions_.front().tracks.end();
//			trackIter++)
//		{
//			(*trackIter)->bCurrentBestSolution = true;
//		}
//	}
//
//	//---------------------------------------------------------
//	// CALCULATE PROBABILITIES
//	//---------------------------------------------------------
//	double solutionLoglikelihoodSum = 0;
//	for (size_t solutionIdx = 0; solutionIdx < queueStGraphSolutions_.size(); solutionIdx++)
//	{
//		stGraphSolution *curSolution = &(queueStGraphSolutions_[solutionIdx]);
//		solutionLoglikelihoodSum += curSolution->logLikelihood;
//		for (size_t trackIdx = 0; trackIdx < curSolution->tracks.size(); trackIdx++)
//		{
//			curSolution->tracks[trackIdx]->GTProb += curSolution->logLikelihood;
//		}
//	}
//
//	// probability of global hypothesis
//	for (size_t solutionIdx = 0; solutionIdx < queueStGraphSolutions_.size(); solutionIdx++)
//	{
//		queueStGraphSolutions_[solutionIdx].probability = queueStGraphSolutions_[solutionIdx].logLikelihood / solutionLoglikelihoodSum;		
//	}
//
//	// global track probability
//	for (size_t trackIdx = 0; trackIdx < queueTracksInWindow_.size(); trackIdx++)
//	{
//		Track3D *curTrack = queueTracksInWindow_[trackIdx];
//		curTrack->GTProb /= solutionLoglikelihoodSum;
//		if (curTrack->tree->maxGTProb < curTrack->GTProb)
//		{
//			curTrack->tree->maxGTProb = curTrack->GTProb;
//		}
//	}
//}
//
//
///************************************************************************
// Method Name: Track3D_Pruning
// Description: 
//	- prune tracks
// Input Arguments:
//	- none
// Return Values:
//	- none
//************************************************************************/
//void CPSNWhere_Associator3D::Track3D_Pruning_GTP(void)
//{
//	// sort
//	std::sort(queueTracksInWindow_.begin(), queueTracksInWindow_.end(), psnTrackGTPandLLDescend);
//
//	//---------------------------------------------------------
//	// TRACK PRUNING
//	//---------------------------------------------------------
//	PSN_TrackSet newQueueTracksInWindow;
//	size_t numTracksInWindow = 0;
//	size_t numTracksPruned = 0;
//	for(PSN_TrackSet::iterator trackIter = queueTracksInWindow_.begin();
//		trackIter != queueTracksInWindow_.end();
//		trackIter++)
//	{
//		if(numTracksInWindow < MAX_TRACK_IN_OPTIMIZATION || (*trackIter)->duration < NUM_FRAME_FOR_CONFIRMATION)
//		{
//			newQueueTracksInWindow.push_back(*trackIter);
//			numTracksInWindow++;
//		}
//		else
//		{
//			(*trackIter)->bValid = false;
//			numTracksPruned++;
//		}
//	}
//	queueTracksInWindow_ = newQueueTracksInWindow;
//
//
//	//---------------------------------------------------------
//	// REPAIRING DATA STRUCTURES
//	//---------------------------------------------------------	
//	this->Track3D_RepairDataStructure();
//
//#ifdef PSN_DEBUG_MODE_
//	printf("[CPSNWhere_Associator3D](Pruing) deleted tracks: %d\n", (int)numTracksPruned);
//#endif
//}
//
//
///************************************************************************
// Method Name: Track3D_Pruning_KBest
// Description: 
//	- prune tracks with K-best solutions
// Input Arguments:
//	- none
// Return Values:
//	- none
//************************************************************************/
//void CPSNWhere_Associator3D::Track3D_Pruning_KBest(void)
//{
//	int numTrackLeft = 0;
//	int numUCTrackLeft = 0;
//	
//	std::sort(queueTracksInWindow_.begin(), queueTracksInWindow_.end(), psnTrackGTPandLLDescend);	
//	for(PSN_TrackSet::iterator trackIter = queueTracksInWindow_.begin();
//		trackIter != queueTracksInWindow_.end();
//		trackIter++)
//	{
//		//if((*trackIter)->tree->timeGeneration + NUM_FRAME_FOR_CONFIRMATION > nCurrentFrameIdx_ && numUCTrackLeft < MAX_UNCONFIRMED_TRACK) 
//		if((*trackIter)->tree->timeGeneration + NUM_FRAME_FOR_CONFIRMATION > nCurrentFrameIdx_) 
//		{ 
//			numUCTrackLeft++;
//			continue; 
//		}
//
//		if(numTrackLeft < MAX_TRACK_IN_OPTIMIZATION && (*trackIter)->GTProb > 0) 
//		{ 	
//			numTrackLeft++;
//			continue;
//		}
//
//		(*trackIter)->bValid = false;
//	}
//}
//
///************************************************************************
// Method Name: Track3D_RepairDataStructure
// Description: 
//	- 
// Input Arguments:
//	- 
// Return Values:
//	- 
//************************************************************************/
//void CPSNWhere_Associator3D::Track3D_RepairDataStructure()
//{
//	PSN_TrackSet newTrackQueue;
//
//	// m_queueActiveTrack
//	for(size_t trackIdx = 0; trackIdx < queueActiveTrack_.size(); trackIdx++)		
//	{
//		Track3D *curTrack = queueActiveTrack_[trackIdx];
//		if(curTrack->bValid){ newTrackQueue.push_back(curTrack); }
//	}
//	queueActiveTrack_ = newTrackQueue;
//	newTrackQueue.clear();
//
//	// m_queueDeactivatedTrack
//	for(size_t trackIdx = 0; trackIdx < queuePausedTrack_.size(); trackIdx++)		
//	{
//		Track3D *curTrack = queuePausedTrack_[trackIdx];
//		if(curTrack->bValid){ newTrackQueue.push_back(curTrack); }
//	}	
//	queuePausedTrack_ = newTrackQueue;
//	newTrackQueue.clear();
//
//	// m_queueTracksInWindow
//	int numTrackPruned = (int)queueTracksInWindow_.size();
//	for(size_t trackIdx = 0; trackIdx < queueTracksInWindow_.size(); trackIdx++)		
//	{
//		Track3D *curTrack = queueTracksInWindow_[trackIdx];
//		if(curTrack->bValid){ newTrackQueue.push_back(curTrack); }
//	}	
//	queueTracksInWindow_ = newTrackQueue;
//	numTrackPruned =- (int)queueTracksInWindow_.size();
//	newTrackQueue.clear();
//
//	// m_queueActiveTrees
//	std::deque<TrackTree*> newTreeQueue;
//	for(size_t treeIdx = 0; treeIdx < queuePtActiveTrees_.size(); treeIdx++)
//	{
//		if(queuePtActiveTrees_[treeIdx]->bSelected)
//		{
//			newTreeQueue.push_back(queuePtActiveTrees_[treeIdx]);
//		}
//	}
//	queuePtActiveTrees_ = newTreeQueue;
//}

/************************************************************************
 Method Name: ComputeEnterProbability
 Description: 
	- calculate entering probability
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
double CPSNWhere_Associator3D::ComputeEnterProbability(std::vector<PSN_Point2D_CamIdx> &vecPointInfos)
{
	float distanceFromBoundary = 0.0;
	for(unsigned int infoIdx = 0; infoIdx < vecPointInfos.size(); infoIdx++)
	{
		PSN_Point2D curPoint = vecPointInfos[infoIdx].first;
		unsigned int curCamIdx = vecPointInfos[infoIdx].second;
		float curDistance = matDistanceFromBoundary_[curCamIdx].at<float>((int)curPoint.y, (int)curPoint.x);

		if(distanceFromBoundary < curDistance)
		{
			distanceFromBoundary = curDistance;
		}
	}
	return distanceFromBoundary <= BOUNDARY_DISTANCE? P_EN_MAX : P_EN_MAX * exp(-(double)(P_EN_DECAY * (distanceFromBoundary - BOUNDARY_DISTANCE)));
}

/************************************************************************
 Method Name: ComputeExitProbability
 Description: 
	- calculate exit probability
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
double CPSNWhere_Associator3D::ComputeExitProbability(std::vector<PSN_Point2D_CamIdx> &vecPointInfos)
{
	float distanceFromBoundary = 0.0;
	for(unsigned int infoIdx = 0; infoIdx < vecPointInfos.size(); infoIdx++)
	{
		PSN_Point2D curPoint = vecPointInfos[infoIdx].first;
		unsigned int curCamIdx = vecPointInfos[infoIdx].second;
		float curDistance = matDistanceFromBoundary_[curCamIdx].at<float>((int)curPoint.y, (int)curPoint.x);

		if(distanceFromBoundary < curDistance)
		{
			distanceFromBoundary = curDistance;
		}
	}
	return distanceFromBoundary <= BOUNDARY_DISTANCE? P_EX_MAX : P_EX_MAX * exp(-(double)(P_EX_DECAY * (distanceFromBoundary - BOUNDARY_DISTANCE)));
}


/************************************************************************
 Method Name: ComputeLinkProbability
 Description: 
	- calculate linking probability
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
double CPSNWhere_Associator3D::ComputeLinkProbability(PSN_Point3D &prePoint, PSN_Point3D &curPoint, unsigned int timeGap)
{
	double distance = (prePoint - curPoint).norm_L2();
	double maxDistance = MAX_MOVING_SPEED * timeGap;
	return distance > maxDistance ? 0.0 : 0.5 * psn::erfc(4.0 * distance / maxDistance - 2.0);
}


/************************************************************************
 Method Name: CheckIncompatibility
 Description: 
	- check incompatibility between two tracks
 Input Arguments:
	- 
 Return Values:
	- whether they are incompatible or not
************************************************************************/
bool CPSNWhere_Associator3D::CheckIncompatibility(Track3D *track1, Track3D *track2)
{
	bool bIncompatible = false;

	// check coupling
	if(track1->tree->id == track2->tree->id)
	{
		bIncompatible = true;
	}
	else
	{
		for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if(0 == track1->tracklet2DIDs[camIdx].size() || 0 == track2->tracklet2DIDs[camIdx].size())
			{
				continue;
			}

			if(track1->tracklet2DIDs[camIdx][0] > track2->tracklet2DIDs[camIdx][0])
			{
				if(track1->tracklet2DIDs[camIdx][0] > track2->tracklet2DIDs[camIdx].back())
				{
					continue;
				}

				for(std::deque<unsigned int>::iterator tracklet1Iter = track2->tracklet2DIDs[camIdx].begin();
					tracklet1Iter != track2->tracklet2DIDs[camIdx].end();
					tracklet1Iter++)
				{
					for(std::deque<unsigned int>::iterator tracklet2Iter = track1->tracklet2DIDs[camIdx].begin();
						tracklet2Iter != track1->tracklet2DIDs[camIdx].end();
						tracklet2Iter++)	
					{
						if(*tracklet1Iter != *tracklet2Iter){ continue; }
						
						bIncompatible = true;
						break;
					}
					if(bIncompatible)
					{
						break;
					}
				}
			}
			else if(track1->tracklet2DIDs[camIdx][0] < track2->tracklet2DIDs[camIdx][0])
			{
				if(track1->tracklet2DIDs[camIdx].back() < track2->tracklet2DIDs[camIdx][0])
				{
					continue;
				}

				for(std::deque<unsigned int>::iterator tracklet1Iter = track1->tracklet2DIDs[camIdx].begin();
					tracklet1Iter != track1->tracklet2DIDs[camIdx].end();
					tracklet1Iter++)
				{
					for(std::deque<unsigned int>::iterator tracklet2Iter = track2->tracklet2DIDs[camIdx].begin();
						tracklet2Iter != track2->tracklet2DIDs[camIdx].end();
						tracklet2Iter++)
					{
						if(*tracklet1Iter != *tracklet2Iter){ continue; }
						
						bIncompatible = true;
						break;
					}
					if(bIncompatible)
					{
						break;
					}
				}
			}
			else
			{
				bIncompatible = true;
				break;
			}

			if(bIncompatible)
			{
				break; 
			}
		}
	}

	// check proximity
	if(!bIncompatible)
	{
		// check overlapping
		if(track1->timeEnd < track2->timeStart || track2->timeEnd < track1->timeStart)
		{
			return bIncompatible;
		}

		unsigned int timeStart = std::max(track1->timeStart, track2->timeStart);
		unsigned int timeEnd = std::min(track1->timeEnd, track2->timeEnd);
		unsigned int track1ReconIdx = timeStart - track1->timeStart;
		unsigned int track2ReconIdx = timeStart - track2->timeStart;
		unsigned int overlapLength = timeEnd - timeStart + 1;
		PSN_Point3D *reconLocation1;
		PSN_Point3D *reconLocation2;
		for(unsigned int reconIdx = 0; reconIdx < overlapLength; reconIdx++)
		{
			reconLocation1 = &(track1->reconstructions[track1ReconIdx].point);
			reconLocation2 = &(track2->reconstructions[track2ReconIdx].point);

			if((*reconLocation1 - *reconLocation2).norm_L2() < MAX_TARGET_PROXIMITY)
			{
				return true;
			}

			track1ReconIdx++;
			track2ReconIdx++;
		}
	}

	return bIncompatible;
}

/************************************************************************
 Method Name: CheckIncompatibility
 Description: 
	- check incompatibility between two tracks
 Input Arguments:
	- 
 Return Values:
	- whether they are incompatible or not
************************************************************************/
bool CPSNWhere_Associator3D::CheckIncompatibility(CTrackletCombination &combi1, CTrackletCombination &combi2)
{
	bool bIncompatible = false;
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(NULL == combi1.get(camIdx) || NULL == combi2.get(camIdx))
		{
			continue;
		}
		if(combi1.get(camIdx) == combi2.get(camIdx))
		{
			bIncompatible = true;
			break;
		}
	}

	return bIncompatible;
}

/************************************************************************
 Method Name: Hypothesis_UpdateHypotheses
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Hypothesis_UpdateHypotheses(PSN_HypothesisSet &inoutUpdatedHypotheses, PSN_TrackSet *newSeedTracks)
{
	//---------------------------------------------------------
	// ADD NEW TRACKS TO RELATED TRACK QUEUE
	//---------------------------------------------------------
	PSN_HypothesisSet newHypothesesSet;	
	for (PSN_HypothesisSet::iterator hypothesisIter = inoutUpdatedHypotheses.begin();
		hypothesisIter != inoutUpdatedHypotheses.end() && newHypothesesSet.size() < K_BEST_SIZE;
		hypothesisIter++)
	{
		if (K_BEST_SIZE <= newHypothesesSet.size()) { break; }
		//if (!(*hypothesisIter).bValid){ continue; }
		stGlobalHypothesis newGlobalHypothesis = (*hypothesisIter);

		// add new branch tracks
		for (PSN_TrackSet::iterator trackIter = (*hypothesisIter).relatedTracks.begin();
			trackIter != (*hypothesisIter).relatedTracks.end();
			trackIter++)
		{		
			for (PSN_TrackSet::iterator childTrackIter = (*trackIter)->childrenTrack.begin();
				childTrackIter != (*trackIter)->childrenTrack.end();
				childTrackIter++)
			{
				if ((*childTrackIter)->bNewTrack)
				{
					newGlobalHypothesis.relatedTracks.push_back(*childTrackIter);
				}
			}					
		}

		// add new seed tracks
		newGlobalHypothesis.relatedTracks.insert(newGlobalHypothesis.relatedTracks.end(), newSeedTracks->begin(), newSeedTracks->end());
		
		// save hypothesis
		newHypothesesSet.push_back(newGlobalHypothesis);
	}
	inoutUpdatedHypotheses = newHypothesesSet;
}

/************************************************************************
 Method Name: Hypothesis_Formation
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Hypothesis_Formation(PSN_HypothesisSet &outBranchHypotheses, PSN_HypothesisSet *existingHypotheses)
{
	//---------------------------------------------------------
	// PROPAGATE GLOBAL HYPOTHESES
	//---------------------------------------------------------
	outBranchHypotheses.clear();
	if (0 == existingHypotheses->size())
	{
		this->Hypothesis_BranchHypotheses(outBranchHypotheses, &this->Track3D_GetWholeCandidateTracks());
	}
	else
	{	
		// solve K times
#pragma omp parallel for
		for (size_t hypothesisIdx = 0; hypothesisIdx < existingHypotheses->size(); hypothesisIdx++)
		{
			this->Hypothesis_BranchHypotheses(
				outBranchHypotheses, 
				&(*existingHypotheses)[hypothesisIdx].relatedTracks,
				&(*existingHypotheses)[hypothesisIdx].selectedTracks);
		}
	}
	if (0 == outBranchHypotheses.size()) { return; }

	//---------------------------------------------------------
	// CALCULATE PROBABILITIES
	//---------------------------------------------------------
	double hypothesesTotalLoglikelihood = 0.0;
	for (size_t solutionIdx = 0; solutionIdx < outBranchHypotheses.size(); solutionIdx++)
	{
		hypothesesTotalLoglikelihood += outBranchHypotheses[solutionIdx].logLikelihood;		
	}

	// probability of global hypothesis and global track probability
	for (size_t hypothesisIdx = 0; hypothesisIdx < outBranchHypotheses.size(); hypothesisIdx++)
	{
		outBranchHypotheses[hypothesisIdx].probability = outBranchHypotheses[hypothesisIdx].logLikelihood / hypothesesTotalLoglikelihood;		
		for (size_t trackIdx = 0; trackIdx < outBranchHypotheses[hypothesisIdx].selectedTracks.size(); trackIdx++)
		{
			outBranchHypotheses[hypothesisIdx].selectedTracks[trackIdx]->GTProb += outBranchHypotheses[hypothesisIdx].probability;
		}
	}
}

/************************************************************************
 Method Name: Hypothesis_BranchHypotheses
 Description: 
	- construct graph with inserted tracks ans solve that graph
 Input Arguments:
	- tracks: related tracks
	- initialselectedTracks: tracks in previous solution (or initial solution)
 Return Values:
	- none
************************************************************************/
#define PSN_GRAPH_SOLUTION_DUPLICATION_RESOLUTION (1.0E-5)
void CPSNWhere_Associator3D::Hypothesis_BranchHypotheses(PSN_HypothesisSet &outBranchHypotheses, PSN_TrackSet *tracks, PSN_TrackSet *initialselectedTracks)
{
#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](Hypothesis_BranchHypotheses) start\n");
	time_t timer_start = clock();
#endif

	if (0 == tracks->size())
	{ 
#ifdef PSN_DEBUG_MODE_ 
		printf("[CPSNWhere_Associator3D](Hypothesis_BranchHypotheses) end with no tracks\n"); 
#endif
		return; 
	}

	//---------------------------------------------------------
	// GRAPH CONSTRUCTION
	//---------------------------------------------------------	
	PSN_Graph *curGraph = new PSN_Graph();
	PSN_VertexSet vertexInGraph;
	PSN_VertexSet vertexInInitialSolution;
	std::deque<PSN_VertexSet> initialSolutionSet;
	std::deque<Track3D*> queueTracksInOptimization;

	for (size_t trackIdx = 0; trackIdx < tracks->size(); trackIdx++)		
	{
		Track3D *curTrack = (*tracks)[trackIdx];

		// TODO: proximity with initial tracks
		double curEnteringCost = curTrack->costEnter;
		if (NULL != initialselectedTracks)
		{
			for (PSN_TrackSet::iterator initialTrackIter = initialselectedTracks->begin();
				initialTrackIter != initialselectedTracks->end();
				initialTrackIter++)
			{
				if ((*initialTrackIter)->bActive) { continue; }
			}
		}		

		// cost	
		curTrack->costTotal = curEnteringCost
							+ curTrack->costReconstruction
							+ curTrack->costLink
							+ curTrack->costExit;

		if (0.0 < curTrack->costTotal) { continue; }

		// for miss penalty
		double unpenaltyCost = 0.0;
		for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if (NULL == curTrack->curTracklet2Ds.get(camIdx)) { continue; }
			unpenaltyCost += log(FP_RATE/(1 - FP_RATE));
		}

		// add vertex
		curTrack->loglikelihood = -(curTrack->costTotal + unpenaltyCost);
		PSN_GraphVertex *curVertex = curGraph->AddVertex(curTrack->loglikelihood);
		vertexInGraph.push_back(curVertex);
		queueTracksInOptimization.push_back(curTrack);

		// find incompatibility constraints
		for (unsigned int compTrackIdx = 0; compTrackIdx < queueTracksInOptimization.size() - 1; compTrackIdx++)
		{			
			if (!CheckIncompatibility(curTrack, queueTracksInOptimization[compTrackIdx]))
			{
				// for MCP
				curGraph->AddEdge(vertexInGraph[compTrackIdx], vertexInGraph.back());
			}
		}

		// initial solution related
		if (NULL == initialselectedTracks)
		{ continue; }

		PSN_TrackSet::iterator findIter = std::find(initialselectedTracks->begin(), initialselectedTracks->end(), curTrack);
		if (initialselectedTracks->end() != findIter)
		{
			vertexInInitialSolution.push_back(curVertex);
		}
	}

	// validate initial solution
	if (NULL != initialselectedTracks)
	{
		initialSolutionSet.push_back(vertexInInitialSolution);
	}

	if (0 == curGraph->Size())
	{
		delete curGraph;
		return;
	}

#ifdef PSN_DEBUG_MODE_
	time_t timer_end = clock();
	double constructionTime = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
	printf("[CPSNWhere_Associator3D](Hypothesis_BranchHypotheses) Vertex:%d, Edge:%d, time:%f\n", curGraph->Size(), curGraph->NumEdge(), constructionTime);
	time_t timer_solver = clock();
#endif	
	
	//---------------------------------------------------------
	// GRAPH SOLVING
	//---------------------------------------------------------
	cGraphSolver_.SetGraph(curGraph);
	cGraphSolver_.SetInitialPoints(initialSolutionSet);
	stGraphSolvingResult *curResult = cGraphSolver_.Solve();

#ifdef PSN_DEBUG_MODE_
	timer_end = clock();
	fCurrentSolvingTime_ = (double)(timer_end - timer_solver) / CLOCKS_PER_SEC;
	size_t numHypothesesBefore = outBranchHypotheses.size();
#endif

	//---------------------------------------------------------
	// SOLUTION CONVERSION AND DUPLICATION CHECK
	//---------------------------------------------------------
	for (size_t solutionIdx = 0; solutionIdx < curResult->vecSolutions.size(); solutionIdx++)
	{
		stGlobalHypothesis newHypothesis;
		newHypothesis.selectedTracks.clear();
		newHypothesis.relatedTracks = *tracks;
		newHypothesis.bValid = true;
		newHypothesis.logLikelihood = curResult->vecSolutions[solutionIdx].second;		
		for (PSN_VertexSet::iterator vertexIter = curResult->vecSolutions[solutionIdx].first.begin();
			vertexIter != curResult->vecSolutions[solutionIdx].first.end();
			vertexIter++)
		{
			newHypothesis.selectedTracks.push_back(queueTracksInOptimization[(*vertexIter)->id]);
		}		
		std::sort(newHypothesis.selectedTracks.begin(), newHypothesis.selectedTracks.end(), psnTrackIDAscendComparator);

		// duplication check
		bool bDuplicated = false;
		for (size_t compSolutionIdx = 0; compSolutionIdx < outBranchHypotheses.size(); compSolutionIdx++)
		{
			if (std::abs(outBranchHypotheses[compSolutionIdx].logLikelihood - newHypothesis.logLikelihood) > PSN_GRAPH_SOLUTION_DUPLICATION_RESOLUTION)
			{ continue; }

			if (outBranchHypotheses[compSolutionIdx].selectedTracks == newHypothesis.selectedTracks)
			{
				bDuplicated = true;
				break;
			}
		}
		if (bDuplicated) { continue; }

		outBranchHypotheses.push_back(newHypothesis);
	}

	//---------------------------------------------------------
	// WRAP-UP
	//---------------------------------------------------------
	delete curGraph;

#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](Hypothesis_BranchHypotheses) solving\n");
	printf(" - num. of solutions: %d\n", outBranchHypotheses.size() - numHypothesesBefore);
	printf(" - time: %f sec\n", fCurrentSolvingTime_);
#endif
}

/************************************************************************
 Method Name: Hypothesis_PruningNScanBack
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::Hypothesis_PruningNScanBack(
	unsigned int nCurrentFrameIdx, 
	unsigned int N, 
	PSN_TrackSet *tracksInWindow, 
	std::deque<stGlobalHypothesis> *ptQueueHypothesis)
{
	if (NULL == ptQueueHypothesis) { return; }
	if (0 >= (*ptQueueHypothesis).size()) { return; }

	// sort by prob. of hypotheses
	std::sort((*ptQueueHypothesis).begin(), (*ptQueueHypothesis).end(), psnSolutionLogLikelihoodDescendComparator);

	// clear all flags of track for pruning (left unconfirmed tracks)
	for (int trackIdx = 0; trackIdx < tracksInWindow->size(); trackIdx++)
	{
		if ((*tracksInWindow)[trackIdx]->tree->timeGeneration + NUM_FRAME_FOR_CONFIRMATION > nCurrentFrameIdx){ continue; }
		(*tracksInWindow)[trackIdx]->bValid = false;
	}

	// branch pruning
	unsigned int nTimeBranchPruning = nCurrentFrameIdx - N;
	PSN_TrackSet *tracksInBestSolution = &(*ptQueueHypothesis).front().selectedTracks;
	Track3D *brachSeedTrack = NULL;
	for (int trackIdx = 0; trackIdx < tracksInBestSolution->size(); trackIdx++)
	{
		brachSeedTrack = TrackTree::FindOldestTrackInBranch((*tracksInBestSolution)[trackIdx], nTimeBranchPruning);

		// DEBUG
		if ((*tracksInBestSolution)[trackIdx] != brachSeedTrack)
		{
			int a = 0;
		}

		TrackTree::SetValidityFlagInTrackBranch(brachSeedTrack, true);
	}

	// get rid of invalid hypothesis
	std::deque<stGlobalHypothesis> existingHypothesis = *ptQueueHypothesis;
	ptQueueHypothesis->clear();
	stGlobalHypothesis *curHypotheis = NULL;
	bool bCurHypothesisValid = true;
	for (int hypothesisIdx = 0; hypothesisIdx < existingHypothesis.size(); hypothesisIdx++)
	{
		curHypotheis = &existingHypothesis[hypothesisIdx];

		// check validity
		bCurHypothesisValid = true;
		for (int trackIdx = 0; trackIdx < curHypotheis->selectedTracks.size(); trackIdx++)
		{
			if (!curHypotheis->selectedTracks[trackIdx]->bValid)
			{ 
				bCurHypothesisValid = false;
				break;
			}
		}
		if (!bCurHypothesisValid) { continue; }

		// update relate track list
		PSN_TrackSet newRelatedTracks;
		for (int trackIdx = 0; trackIdx < curHypotheis->relatedTracks.size(); trackIdx++)
		{
			if (!curHypotheis->relatedTracks[trackIdx]->bValid) { continue; }
			newRelatedTracks.push_back(curHypotheis->relatedTracks[trackIdx]);
		}
		curHypotheis->relatedTracks = newRelatedTracks;

		// save valid hypothesis
		ptQueueHypothesis->push_back(*curHypotheis);
	}
}

/************************************************************************
 Method Name: ResultWithCandidateTracks
 Description: 
	- generate track result with candidate tracks
 Input Arguments:
	- 
 Return Values:
	- stTrack3DResult: a struct of result
************************************************************************/
stTrack3DResult CPSNWhere_Associator3D::ResultWithTracks(PSN_TrackSet *trackSet, unsigned int nFrameIdx, double fProcessingTime)
{
	stTrack3DResult result3D;
	result3D.frameIdx = nFrameIdx;
	result3D.processingTime = fProcessingTime;
	if (0 == trackSet->size()) { return result3D; }	

	for(PSN_TrackSet::iterator trackIter = trackSet->begin();
		trackIter != trackSet->end();
		trackIter++)
	{
		if ((*trackIter)->timeEnd < nFrameIdx) { continue; }

		Track3D *curTrack = *trackIter;
		stObject3DInfo newObject;

		// ID for visualization
		bool bIDNotFound = true;
		for (size_t pairIdx = 0; pairIdx < queuePairTreeIDToVisualizationID_.size(); pairIdx++)
		{
			if (queuePairTreeIDToVisualizationID_[pairIdx].first == curTrack->tree->id)
			{
				newObject.id = queuePairTreeIDToVisualizationID_[pairIdx].second;
				bIDNotFound = false;
				break;
			}
		}
		if (bIDNotFound)
		{
			// DEBUG
			//newObject.id = this->m_nNextVisualizationID++;
			newObject.id = curTrack->id;
			queuePairTreeIDToVisualizationID_.push_back(std::make_pair(curTrack->tree->id, newObject.id));
		}
		
		unsigned numPoint = 0;
		for (std::deque<stReconstruction>::reverse_iterator pointIter = curTrack->reconstructions.rbegin();
			pointIter != curTrack->reconstructions.rend();
			pointIter++)
		{
			numPoint++;
			newObject.recentPoints.push_back((*pointIter).point);			
			if (DISP_TRAJECTORY3D_LENGTH < numPoint) { break; }

			for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				//newObject.point3DBox[camIdx] = this->GetHuman3DBox((*pointIter).point, 500, camIdx);				
				newObject.recentPoint2Ds[camIdx].push_back(this->WorldToImage((*pointIter).point, camIdx));

				if (1 < numPoint) { continue; }
				if (NULL == curTrack->curTracklet2Ds.get(camIdx))
				{
					PSN_Rect curRect(0.0, 0.0, 0.0, 0.0);
					PSN_Point3D topCenterInWorld((*pointIter).point);
					topCenterInWorld.z = 1700;
					PSN_Point2D bottomCenterInImage = this->WorldToImage((*pointIter).point, camIdx);
					PSN_Point2D topCenterInImage = this->WorldToImage(topCenterInWorld, camIdx);
					
					double height = (topCenterInImage - bottomCenterInImage).norm_L2();
					curRect.x = bottomCenterInImage.x - height * 2.5 / 17.0;
					curRect.y = topCenterInImage.y;
					curRect.w = height * 5.0 / 17.0;
					curRect.h = bottomCenterInImage.y - topCenterInImage.y;
					
					newObject.rectInViews[camIdx] = curRect;
					newObject.bVisibleInViews[camIdx] = false;
				}
				else
				{
					newObject.rectInViews[camIdx] = curTrack->curTracklet2Ds.get(camIdx)->rects.back();
					newObject.bVisibleInViews[camIdx] = true;
				}
			}
		}
		result3D.object3DInfo.push_back(newObject);
	}

	return result3D;
}

/************************************************************************
 Method Name: printBatchResult
 Description: 
	- print out information of tracks
 Input Arguments:
	- queueTracks: seletect tracks for print out
	- strFilePathAndName: output file path and name
	- bAppend: option for appending
 Return Values:
	- void
************************************************************************/
void CPSNWhere_Associator3D::PrintTracks(std::deque<Track3D*> &queueTracks, char *strFilePathAndName, bool bAppend)
{
	if(0 == queueTracks.size())
	{
		return;
	}

	char strPrint[256];
	try
	{
		FILE *fp;
		if(bAppend)
		{
			fopen_s(&fp, strFilePathAndName, "a");
		}
		else
		{
			fopen_s(&fp, strFilePathAndName, "w");
			sprintf_s(strPrint, "numCamera:%d\n", (int)NUM_CAM);
			fprintf(fp, strPrint);
			sprintf_s(strPrint, "numTracks:%d\n", (int)queueTracks.size());
			fprintf(fp, strPrint);
		}

		for(std::deque<Track3D*>::iterator trackIter = queueTracks.begin();
			trackIter != queueTracks.end();
			trackIter++)
		{
			Track3D *curTrack = *trackIter;

			sprintf_s(strPrint, "{\n\tid:%d\n\ttreeID:%d\n", (int)curTrack->id, (int)curTrack->tree->id);
			fprintf(fp, strPrint);

			sprintf_s(strPrint, "\tnumReconstructions:%d\n\ttimeStart:%d\n\ttimeEnd:%d\n\ttimeGeneration:%d\n", (int)curTrack->reconstructions.size(), (int)curTrack->timeStart, (int)curTrack->timeEnd, (int)curTrack->timeGeneration);
			fprintf(fp, strPrint);

			sprintf_s(strPrint, "\ttrackleIDs:\n\t{\n");
			fprintf(fp, strPrint);
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				if(0 == curTrack->tracklet2DIDs[camIdx].size())
				{
					fprintf(fp, "\t\tnumTracklet:0,[]\n");
					continue;
				}

				sprintf_s(strPrint, "\t\tnumTracklet:%d,[", (int)curTrack->tracklet2DIDs[camIdx].size());
				fprintf(fp, strPrint);
				for(unsigned int trackletIDIdx = 0; trackletIDIdx < curTrack->tracklet2DIDs[camIdx].size(); trackletIDIdx++)
				{
					sprintf_s(strPrint, "%d", (int)curTrack->tracklet2DIDs[camIdx][trackletIDIdx]);
					fprintf(fp, strPrint);
					if(curTrack->tracklet2DIDs[camIdx].size() - 1 != trackletIDIdx)
					{
						fprintf(fp, ",");
					}
					else
					{
						fprintf(fp, "]\n");
					}
				}
			}
			fprintf(fp, "\t}\n");

			sprintf_s(strPrint, "\ttotalCost:%e\n", curTrack->costTotal);
			fprintf(fp, strPrint);

			sprintf_s(strPrint, "\treconstructionCost:%e\n", curTrack->costReconstruction);
			fprintf(fp, strPrint);

			sprintf_s(strPrint, "\tlinkCost:%e\n", curTrack->costLink);
			fprintf(fp, strPrint);

			sprintf_s(strPrint, "\tinitCost:%e\n", curTrack->costEnter);
			fprintf(fp, strPrint);

			sprintf_s(strPrint, "\ttermCost:%e\n", curTrack->costExit);
			fprintf(fp, strPrint);

			fprintf(fp, "\treconstructions:\n\t{\n");
			for(std::deque<stReconstruction>::iterator pointIter = curTrack->reconstructions.begin();
				pointIter != curTrack->reconstructions.end();
				pointIter++)
			{
				sprintf_s(strPrint, "\t\t%d:(%f,%f,%f)\n", (int)(*pointIter).bIsMeasurement, (*pointIter).point.x, (*pointIter).point.y, (*pointIter).point.z);
				fprintf(fp, strPrint);
			}

			sprintf_s(strPrint, "\t}\n}\n");
			fprintf(fp, strPrint);
		}
		
		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](PrintTracks) cannot open file! error code %d\n", dwError);
		return;
	}
}


/************************************************************************
 Method Name: FilePrintCurrentTrackTrees
 Description: 
	- print out tree structure of current tracks
 Input Arguments:
	- strFilePathAndName: output file path and name
 Return Values:
	- void
************************************************************************/
void CPSNWhere_Associator3D::FilePrintCurrentTrackTrees(const char *strFilePath)
{
	try
	{
		FILE *fp;
		fopen_s(&fp, strFilePath, "w");

		std::deque<unsigned int> queueNodes;
		for(std::list<TrackTree>::iterator treeIter = listTrackTree_.begin();
			treeIter != listTrackTree_.end();
			treeIter++)
		{
			if(0 == (*treeIter).tracks.size())
			{
				continue;
			}

			Track3D* curTrack = (*treeIter).tracks.front();
			queueNodes.push_back(0);
			TrackTree::MakeTreeNodesWithChildren(curTrack->childrenTrack, (unsigned int)queueNodes.size(), queueNodes);
		}

		fprintf(fp, "nodeLength:%d\n[", (int)queueNodes.size());
		for(std::deque<unsigned int>::iterator idxIter = queueNodes.begin();
			idxIter != queueNodes.end();
			idxIter++)
		{
			fprintf(fp, "%d", (int)(*idxIter));
			if(idxIter != queueNodes.end() - 1)
			{
				fprintf(fp, ",");
			}
		}
		fprintf(fp, "]");
		
		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](PrintTracks) cannot open file! error code %d\n", dwError);
		return;
	}
}


/************************************************************************
 Method Name: FilePrintDefferedResult
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::FilePrintDefferedResult(void)
{
	FILE *fp;
	try
	{		
		fopen_s(&fp, strDefferedResultFileName_, "w");

		for(std::deque<stTrack3DResult>::iterator resultIter = queueDeferredTrackingResult_.begin();
			resultIter != queueDeferredTrackingResult_.end();
			resultIter++)
		{
			fprintf_s(fp, "{\n\tframeIndex:%d\n\tnumObjects:%d\n", (int)(*resultIter).frameIdx, (int)(*resultIter).object3DInfo.size());
			for(std::vector<stObject3DInfo>::iterator objectIter = (*resultIter).object3DInfo.begin();
				objectIter != (*resultIter).object3DInfo.end();
				objectIter++)
			{
				PSN_Point3D curPoint = (*objectIter).recentPoints.back();
				fprintf_s(fp, "\t{id:%d,position:(%f,%f,%f)}\n", (int)(*objectIter).id, (float)curPoint.x, (float)curPoint.y, (float)curPoint.z);
			}
			fprintf_s(fp, "}\n");
		}
		
		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](PrintTracks) cannot open file! error code %d\n", dwError);
		return;
	}
}


/************************************************************************
 Method Name: FilePrintDefferedResult
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::FilePrintInstantResult(void)
{
	FILE *fp;
	try
	{		
		fopen_s(&fp, strInstantResultFileName_, "w");

		for(std::deque<stTrack3DResult>::iterator resultIter = queueTrackingResult_.begin();
			resultIter != queueTrackingResult_.end();
			resultIter++)
		{
			fprintf_s(fp, "{\n\tframeIndex:%d\n\tnumObjects:%d\n", (int)(*resultIter).frameIdx, (int)(*resultIter).object3DInfo.size());
			for(std::vector<stObject3DInfo>::iterator objectIter = (*resultIter).object3DInfo.begin();
				objectIter != (*resultIter).object3DInfo.end();
				objectIter++)
			{
				PSN_Point3D curPoint = (*objectIter).recentPoints.back();
				fprintf_s(fp, "\t{id:%d,position:(%f,%f,%f)}\n", (int)(*objectIter).id, (float)curPoint.x, (float)curPoint.y, (float)curPoint.z);
			}
			fprintf_s(fp, "}\n");
		}
		
		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](PrintTracks) cannot open file! error code %d\n", dwError);
		return;
	}
}


/************************************************************************
 Method Name: SaveDefferedResult
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::SaveDefferedResult(unsigned int deferredLength)
{
	if(nCurrentFrameIdx_ < deferredLength)
	{
		return;
	}

	for(; nLastPrintedDeferredResultFrameIdx_ + deferredLength <= nCurrentFrameIdx_; nLastPrintedDeferredResultFrameIdx_++)
	{
		stTrack3DResult newResult;
		newResult.frameIdx = nLastPrintedDeferredResultFrameIdx_;
		newResult.processingTime = 0.0;
		newResult.object3DInfo.clear();

		for(std::deque<Track3D*>::iterator trackIter = queueTracksInBestSolution_.begin();
			trackIter != queueTracksInBestSolution_.end();
			trackIter++)
		{
			if((*trackIter)->timeEnd < nLastPrintedDeferredResultFrameIdx_ || (*trackIter)->timeStart > nLastPrintedDeferredResultFrameIdx_)
			{
				continue;
			}

			unsigned int curPosIdx = nLastPrintedDeferredResultFrameIdx_ - (*trackIter)->timeStart;

			stObject3DInfo newObjectInfo;
			newObjectInfo.id = (*trackIter)->tree->id;
			newObjectInfo.recentPoints.push_back((*trackIter)->reconstructions[curPosIdx].point);

			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				if(NULL == (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx))
				{
					newObjectInfo.bVisibleInViews[camIdx] = false;
					newObjectInfo.rectInViews[camIdx] = PSN_Rect(0, 0, 0, 0);
				}
				else
				{
					newObjectInfo.bVisibleInViews[camIdx] = true;
					unsigned int rectIdx = nLastPrintedDeferredResultFrameIdx_ - (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx)->timeStart;
					newObjectInfo.rectInViews[camIdx] = (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx)->rects[rectIdx];					
				}
			}

			newResult.object3DInfo.push_back(newObjectInfo);
		}
		queueDeferredTrackingResult_.push_back(newResult);
	}
}


/************************************************************************
 Method Name: SaveInstantResult
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::SaveInstantResult(void)
{
	stTrack3DResult newResult;
	newResult.frameIdx = nCurrentFrameIdx_;
	newResult.processingTime = fCurrentProcessingTime_;
	newResult.object3DInfo.clear();

	for(std::deque<Track3D*>::iterator trackIter = queueTracksInBestSolution_.begin();
		trackIter != queueTracksInBestSolution_.end();
		trackIter++)
	{
		if((*trackIter)->timeEnd < newResult.frameIdx || (*trackIter)->timeStart > newResult.frameIdx)
		{
			continue;
		}

		unsigned int curPosIdx = newResult.frameIdx - (*trackIter)->timeStart;

		stObject3DInfo newObjectInfo;
		newObjectInfo.id = (*trackIter)->tree->id;
		newObjectInfo.recentPoints.push_back((*trackIter)->reconstructions[curPosIdx].point);

		for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if(NULL == (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx))
			{
				newObjectInfo.bVisibleInViews[camIdx] = false;
				newObjectInfo.rectInViews[camIdx] = PSN_Rect(0, 0, 0, 0);
			}
			else
			{
				newObjectInfo.bVisibleInViews[camIdx] = true;
				unsigned int rectIdx = newResult.frameIdx - (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx)->timeStart;
				newObjectInfo.rectInViews[camIdx] = (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx)->rects[rectIdx];					
			}
		}

		newResult.object3DInfo.push_back(newObjectInfo);
	}

	queueTrackingResult_.push_back(newResult);
}


///************************************************************************
// Method Name: FilePrintForAnalysis
// Description: 
//	- 
// Input Arguments:
//	- 
// Return Values:
//	- 
//************************************************************************/
//void CPSNWhere_Associator3D::FilePrintForAnalysis(void)
//{
//	try
//	{		
//		FILE *fp;
//
//		char strPrint[256];
//		sprintf_s(strPrint, "logs/analysis/%04d.txt", nCurrentFrameIdx_);
//		fopen_s(&fp, strPrint, "w");
//
//		fprintf_s(fp, "frame:%d\nnumTracks:%d\n", (int)nCurrentFrameIdx_, (int)queueTracksInWindow_.size());		
//
//		//---------------------------------------------------------
//		// PRINT TRACKS
//		//---------------------------------------------------------
//		for(PSN_TrackSet::iterator trackIter = queueTracksInWindow_.begin();
//			trackIter != queueTracksInWindow_.end();
//			trackIter++)
//		{
//			Track3D *curTrack = *trackIter;
//
//			fprintf_s(fp, "{\n\tid:%d\n\ttreeID:%d\n", (int)curTrack->id, (int)curTrack->tree->id);			
//			fprintf_s(fp, "\tnumReconstructions:%d\n\ttimeStart:%d\n\ttimeEnd:%d\n", (int)curTrack->reconstructions.size(), (int)curTrack->timeStart, (int)curTrack->timeEnd);
//			fprintf_s(fp, "\ttrackleIDs:\n\t{\n");			
//			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
//			{
//				if(0 == curTrack->tracklet2DIDs[camIdx].size())
//				{
//					fprintf_s(fp, "\t\tnumTracklet:0,[]\n");
//					continue;
//				}
//
//				fprintf_s(fp, "\t\tnumTracklet:%d,[", (int)curTrack->tracklet2DIDs[camIdx].size());				
//				for(unsigned int trackletIDIdx = 0; trackletIDIdx < curTrack->tracklet2DIDs[camIdx].size(); trackletIDIdx++)
//				{
//					fprintf_s(fp, "%d", (int)curTrack->tracklet2DIDs[camIdx][trackletIDIdx]);					
//					if(curTrack->tracklet2DIDs[camIdx].size() - 1 != trackletIDIdx)
//					{
//						fprintf_s(fp, ",");
//					}
//					else
//					{
//						fprintf_s(fp, "]\n");
//					}
//				}
//			}
//			fprintf_s(fp, "\t}\n");
//			fprintf_s(fp, "\ttotalCost:%e\n", curTrack->costTotal);
//			fprintf_s(fp, "\tinitCost:%e\n", curTrack->costEnter);
//			fprintf_s(fp, "\ttermCost:%e\n", curTrack->costExit);
//			fprintf_s(fp, "\treconstructionCost:%e\n", curTrack->costReconstruction);
//			fprintf_s(fp, "\tlinkCost:%e\n", curTrack->costLink);
//
//			if(curTrack->timeEnd >= nCurrentFrameIdx_)
//			{
//				fprintf_s(fp, "\tcurrentReconstructionCost:%e\n", curTrack->reconstructions.back().costReconstruction);
//				fprintf_s(fp, "\tcurrentLinkCost:%e\n", curTrack->reconstructions.back().costLink);			
//				fprintf_s(fp, "\tcurrentReconstruction:(%f,%f,%f)\n", curTrack->reconstructions.back().point.x, curTrack->reconstructions.back().point.y, curTrack->reconstructions.back().point.z);			
//			}
//			fprintf_s(fp, "}\n");
//		}
//
//		
//		//---------------------------------------------------------
//		// PRINT SOLUTIONS
//		//---------------------------------------------------------
//		fprintf_s(fp, "numSolution:%d\n", (int)queueStGraphSolutions_.size());		
//		for(std::vector<stGraphSolution>::iterator solutionIter = queueStGraphSolutions_.begin();
//			solutionIter != queueStGraphSolutions_.end();
//			solutionIter++)
//		{
//			stGraphSolution *curSolution = &(*solutionIter);
//
//			fprintf_s(fp, "{\n\ttracks:(%d){", (int)curSolution->tracks.size());
//			for(PSN_TrackSet::iterator trackIter = curSolution->tracks.begin();
//				trackIter != curSolution->tracks.end();
//				trackIter++)
//			{
//				fprintf_s(fp, "%d,", (int)(*trackIter)->id);				
//			}
//
//			fprintf_s(fp, "}\n\tloglikelihood:%e\n", curSolution->logLikelihood);
//			fprintf_s(fp, "\tprobability:%e\n}\n", curSolution->probability);			
//		}		
//		
//		fclose(fp);
//	}
//	catch(DWORD dwError)
//	{
//		printf("[ERROR](FilePrintForAnalysis) cannot open file! error code %d\n", dwError);
//		return;
//	}
//}


/************************************************************************
 Method Name: IndexCombination
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
std::deque<std::vector<unsigned int>> CPSNWhere_Associator3D::IndexCombination(std::deque<std::deque<unsigned int>> &inputIndexDoubleArray, size_t curLevel, std::deque<std::vector<unsigned int>> curCombination)
{
	if(0 == curLevel)
	{
		std::deque<std::vector<unsigned int>> newCombinations;
		for(unsigned int indexIdx = 0; indexIdx < (unsigned int)inputIndexDoubleArray[0].size(); indexIdx++)
		{
			std::vector<unsigned int> curIndex(inputIndexDoubleArray.size(), 0);
			curIndex[0] = indexIdx;
			newCombinations.push_back(curIndex);
		}
		return IndexCombination(inputIndexDoubleArray, 1, newCombinations);
	}
	else if(inputIndexDoubleArray.size() <= curLevel)
	{
		return curCombination;
	}

	std::deque<std::vector<unsigned int>> newCombinations;
	for(unsigned int indexIdx = 0; indexIdx < (unsigned int)inputIndexDoubleArray[curLevel].size(); indexIdx++)
	{
		for(size_t combiIdx = 0; combiIdx < curCombination.size(); combiIdx++)
		{
			std::vector<unsigned int> newIndex = curCombination[combiIdx];
			newIndex[curLevel] = indexIdx;
			newCombinations.push_back(newIndex);
		}
	}
	return IndexCombination(inputIndexDoubleArray, curLevel + 1, newCombinations);
}

//()()
//('')HAANJU.YOO
