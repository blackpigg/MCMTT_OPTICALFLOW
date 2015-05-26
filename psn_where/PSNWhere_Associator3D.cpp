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
#include "PSNWhere_SGSmooth.h"

/////////////////////////////////////////////////////////////////////////
// PARAMETERS (unit: mm)
/////////////////////////////////////////////////////////////////////////

// optimization related
#define PROC_WINDOW_SIZE (5)
#define GTP_THRESHOLD (0.0001)
#define MAX_TRACK_IN_OPTIMIZATION (2000)
#define MAX_TRACK_IN_UNCONFIRMED_TREE (2)
#define MAX_UNCONFIRMED_TRACK (50)
#define NUM_FRAME_FOR_GTP_CHECK (3)
#define NUM_FRAME_FOR_CONFIRMATION (3)
#define K_BEST_SIZE (100)
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
#define MIN_TARGET_PROXIMITY (200)

#define DEFAULT_HEIGHT (1700.00)
#define WIDTH_DIFF_RATIO_THRESHOLD (0.5)
#define CONSIDER_SENSITIVITY (false)

// linking related
#define MIN_LINKING_PROBABILITY (1.0E-6)
#define MAX_TIME_JUMP (9)

#define MAX_GROUNDING_HEIGHT (100)
#define COST_RGB_MIN_DIST (0.2)
#define COST_RGB_COEF (100)
#define COST_RGB_DECAY (0.1)
#define COST_TRACKLET_LINK_MIN_DIST (1500.0)
#define COST_TRACKLET_LINK_COEF (0.1)

// probability related
#define MIN_CONSTRUCT_PROBABILITY (0.01)
#define FP_RATE (0.05)
#define FN_RATE (0.1)

// enter/exit related
#define ENTER_PENALTY_FREE_LENGTH (2)
#define BOUNDARY_DISTANCE (700.0)
#define P_EN_MAX (1.0E-3)
#define P_EX_MAX (1.0E-6)
#define P_EN_DECAY (1.0E-3)
#define P_EX_DECAY (1.0E-3)
#define COST_EN_MAX (400.0)
#define COST_EX_MAX (400.0)

// calibration related
#define E_DET (3)
#define E_CAL (500)

// dynamic related
#define KALMAN_PROCESSNOISE_SIG (1.0E-5)
#define KALMAN_MEASUREMENTNOISE_SIG (1.0E-5)
#define KALMAN_POSTERROR_COV (0.1)
#define KALMAN_CONFIDENCE_LEVEN (9)
#define VELOCITY_LEARNING_RATE (0.9)
#define DATASET_FRAME_RATE (7.0)
//#define MAX_MOVING_SPEED (5000.0 / DATASET_FRAME_RATE)
#define MAX_MOVING_SPEED (900.0)
#define MIN_MOVING_SPEED (100.0)

// appearance reltaed
#define IMG_PATCH_WIDTH (20)
#define NUM_BINS_RGB_HISTOGRAM (16)

//const unsigned int arrayCamCombination[8] = {0, 2, 2, 1, 1, 3, 3, 0};
// smoothing related
#define MIN_SMOOTHING_LENGTH (SGS_DEFAULT_SPAN / 2)
std::vector<Qset> precomputedQsets;

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
	if (track1->GTProb > track2->GTProb) { return true; }
	else if (track1->GTProb == track2->GTProb)
	{
		if (track1->duration < track2->duration) { return true; }
	}
	return false;
}

bool psnTrackGTPandLLDescend(const Track3D *track1, const Track3D *track2)
{
	// higher GTProb, lower cost!
	if (track1->GTProb > track2->GTProb) { return true; }
	else if (track1->GTProb == track2->GTProb)
	{
		if (track1->loglikelihood > track2->loglikelihood) {	return true; }
	}
	return false;
}

bool psnTrackGPandLengthComparator(const Track3D *track1, const Track3D *track2)
{
	if (track1->duration > NUM_FRAME_FOR_CONFIRMATION && track2->duration > NUM_FRAME_FOR_CONFIRMATION)
	{
		if (track1->GTProb > track2->GTProb) { return true; }
		else if (track1->GTProb < track2->GTProb) { return false; }
		if (track1->duration > track2->duration) { return true; }
		return false;
	}
	else if (track1->duration >= NUM_FRAME_FOR_CONFIRMATION && track2->duration < NUM_FRAME_FOR_CONFIRMATION) { return false; }
	return true;
}

//bool psnTreeNumMeasurementDescendComparator(const TrackTree *tree1, const TrackTree *tree2)
//{
//	return tree1->numMeasurements > tree2->numMeasurements;
//}

bool psnSolutionLogLikelihoodDescendComparator(const stGlobalHypothesis &solution1, const stGlobalHypothesis &solution2)
{
	return solution1.logLikelihood > solution2.logLikelihood;
}

//bool psnUnconfirmedTrackGTPLengthDescendComparator(const TrackTree *tree1, const TrackTree *tree2)
//{
//	if(tree1->maxGTProb > tree1->maxGTProb)
//	{
//		return true;
//	}	
//	if(tree1->maxGTProb < tree1->maxGTProb)
//	{
//		return false;
//	}
//	if(tree1->timeGeneration < tree2->timeGeneration)
//	{
//		return true;
//	}
//	return false;
//}


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
	bSnapshotReaded_ = false;

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

	// camera model and field of view
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		// camera model
		cCamModel_[camIdx] = vecStCalibInfo[camIdx]->cCamModel;
		matProjectionSensitivity_[camIdx] = vecStCalibInfo[camIdx]->matProjectionSensitivity.clone();
		matDistanceFromBoundary_[camIdx] = vecStCalibInfo[camIdx]->matDistanceFromBoundary.clone();

		// field of view (clockwise)
		pFieldOfView[camIdx][0] = this->ImageToWorld(PSN_Point2D(0.0, 0.0), 0.0, camIdx);
		pFieldOfView[camIdx][1] = this->ImageToWorld(PSN_Point2D((double)cCamModel_[camIdx].width()-1, 0.0), 0.0, camIdx);
		pFieldOfView[camIdx][2] = this->ImageToWorld(PSN_Point2D((double)cCamModel_[camIdx].width()-1, (double)cCamModel_[camIdx].height()-1), 0.0, camIdx);
		pFieldOfView[camIdx][3] = this->ImageToWorld(PSN_Point2D(0.0, (double)cCamModel_[camIdx].height()-1), 0.0, camIdx);
	}
	
	// evaluation
	this->m_cEvaluator.Initialize(datasetPath);
	this->m_cEvaluator_Instance.Initialize(datasetPath);

	// smoothing related
	for (int windowSize = 1; windowSize <= SGS_DEFAULT_SPAN; windowSize++)
	{
		precomputedQsets.push_back(CPSNWhere_SGSmooth::CalculateQ(windowSize));
	}

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

	if (!bInit_) { return; }

#ifdef SAVE_SNAPSHOT_
	this->SaveSnapshot(SNAPSHOT_PATH);
#endif

	/////////////////////////////////////////////////////////////////////////////
	// PRINT RESULT
	/////////////////////////////////////////////////////////////////////////////
#ifdef PSN_PRINT_LOG_
	//// candidate tracks
	//this->PrintTracks(queuePausedTrack_, strTrackLogFileName_, true);
	//this->PrintTracks(queueActiveTrack_, strTrackLogFileName_, true);
#endif

	// tracks in solutions
	std::deque<Track3D*> queueResultTracks;
	//for(std::list<Track3D>::iterator trackIter = listResultTrack3D_.begin();
	//	trackIter != listResultTrack3D_.end();
	//	trackIter++)
	//{
	//	queueResultTracks.push_back(&(*trackIter));
	//}
	for (std::deque<Track3D*>::iterator trackIter = queueTracksInBestSolution_.begin();
		trackIter != queueTracksInBestSolution_.end();
		trackIter++)
	{
		queueResultTracks.push_back(*trackIter);
	}


#ifdef PSN_PRINT_LOG_
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
	//this->SaveDefferedResult(0);
	this->PrintResult(strInstantResultFileName_, &queueTrackingResult_);
	this->PrintResult(strDefferedResultFileName_, &queueDeferredTrackingResult_);

	/////////////////////////////////////////////////////////////////////////////
	// EVALUATION
	/////////////////////////////////////////////////////////////////////////////
#ifndef LOAD_SNAPSHOT_
	printf("[EVALUATION] deferred result\n");
	for (unsigned int timeIdx = nCurrentFrameIdx_ - PROC_WINDOW_SIZE + 2; timeIdx <= nCurrentFrameIdx_; timeIdx++)
	{
		this->m_cEvaluator.SetResult(queueTracksInBestSolution_, timeIdx);
	}
	this->m_cEvaluator.Evaluate();
	this->m_cEvaluator.PrintResultToConsole();
	this->m_cEvaluator.PrintResultToFile("data/evaluation_deferred.txt");
	this->m_cEvaluator.PrintResultMatrix("data/result_matrix_deferred.txt");

	printf("[EVALUATION] instance result\n");	
	this->m_cEvaluator_Instance.Evaluate();
	this->m_cEvaluator_Instance.PrintResultToConsole();
	this->m_cEvaluator_Instance.PrintResultToFile("data/evaluation_instance.txt");
	this->m_cEvaluator_Instance.PrintResultMatrix("data/result_matrix_instance.txt");
#endif

	/////////////////////////////////////////////////////////////////////////////
	// FINALIZE
	/////////////////////////////////////////////////////////////////////////////

	// clean-up camera model
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		//delete cCamModel_[camIdx];
		matProjectionSensitivity_[camIdx].release();
		matDistanceFromBoundary_[camIdx].release();
	}

	fCurrentProcessingTime_ = 0.0;
	nCurrentFrameIdx_ = 0;

	// 2D tracklet related
	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		vecTracklet2DSet_[camIdx].activeTracklets.clear();
		for (std::list<stTracklet2D>::iterator trackletIter = vecTracklet2DSet_[camIdx].tracklets.begin();
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

#ifdef LOAD_SNAPSHOT_
	if (!bSnapshotReaded_) { bSnapshotReaded_ = this->LoadSnapshot(SNAPSHOT_PATH); }
	if (bSnapshotReaded_ && frameIdx <= (int)nCurrentFrameIdx_)
	{
		stTrack3DResult curResult;
		if (frameIdx < queueTrackingResult_.size()) { curResult = queueTrackingResult_[frameIdx]; }
		return curResult;
	}
#endif

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
	if (bInitiationPenaltyFree_ && nCountForPenalty_++ > ENTER_PENALTY_FREE_LENGTH) { bInitiationPenaltyFree_ = false; }

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

	//if (80 == nCurrentFrameIdx_)
	//{
	//	this->PrintCurrentTrackTrees("data/trees.txt");
	//}

	// post-pruning
	this->Hypothesis_PruningNScanBack(nCurrentFrameIdx_, PROC_WINDOW_SIZE, &queueTracksInWindow_, &queueCurrGlobalHypotheses_);
	this->Hypothesis_PruningTrackWithGTP(nCurrentFrameIdx_, MAX_TRACK_IN_OPTIMIZATION, &queueTracksInWindow_);
	this->Hypothesis_RefreshHypotheses(queueCurrGlobalHypotheses_);

	//if (80 == nCurrentFrameIdx_)
	//{
	//	this->PrintCurrentTrackTrees("data/trees_afterPruning.txt");
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

	// instance result
	stTrack3DResult currentResult = this->ResultWithTracks(&queueTracksInBestSolution_, frameIdx, processingTime);
	queueTrackingResult_.push_back(currentResult);

	// deferred result
	if (frameIdx + 1 >= PROC_WINDOW_SIZE)
	{
		queueDeferredTrackingResult_.push_back(this->ResultWithTracks(&queueTracksInBestSolution_, frameIdx + 1 - PROC_WINDOW_SIZE, processingTime));
	}
	
	//this->SaveInstantResult();
	//this->SaveDefferedResult(PROC_WINDOW_SIZE);

	/////////////////////////////////////////////////////////////////////////////
	// EVALUATION
	/////////////////////////////////////////////////////////////////////////////
	int timeDeferred = (int)nCurrentFrameIdx_ - PROC_WINDOW_SIZE + 1;
	if (0 <= timeDeferred)
	{
		this->m_cEvaluator.SetResult(queueTracksInBestSolution_, timeDeferred);
	}
	this->m_cEvaluator_Instance.SetResult(queueTracksInBestSolution_, nCurrentFrameIdx_);

	/////////////////////////////////////////////////////////////////////////////
	// WRAP-UP
	/////////////////////////////////////////////////////////////////////////////

	// memory clean-up
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
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

#ifdef PSN_PRINT_LOG_
	// PRINT LOG
	std::string strLog = "";
	strLog += std::to_string(nCurrentFrameIdx_) + ",";
	strLog += std::to_string(nCountTrackInOptimization_) + ",";
	strLog += std::to_string(nCountUCTrackInOptimization_) + ",";
	strLog += std::to_string(fCurrentProcessingTime_) + ",";
	strLog += std::to_string(fCurrentSolvingTime_) + ",";
	strLog += std::to_string(nCurrentFrameIdx_) + ",";
	//sprintf_s(strLog, "%d,%d,%d,%f,%f", 
	//nCurrentFrameIdx_, 
	//nCountTrackInOptimization_,
	//nCountUCTrackInOptimization_,
	//fCurrentProcessingTime_,
	//fCurrentSolvingTime_);
	strLog += "\n";
	psn::printLog(strLogFileName_, strLog);
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
	for (unsigned int vertexIdx = 0; vertexIdx < 8; vertexIdx++)
	{
		PSN_Point2D curPoint = this->WorldToImage(PSN_Point3D(ptHeadCenter.x + offsetX[vertexIdx], ptHeadCenter.y + offsetY[vertexIdx], z[vertexIdx]), camIdx);
		resultArray.push_back(curPoint);
		if (curPoint.onView(cCamModel_[camIdx].width(), cCamModel_[camIdx].height())) { numPointsOnView++; }
	}

	// if there is no point can be seen from the view, clear a vector of point
	if (0 == numPointsOnView) { resultArray.clear(); }

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

		for (unsigned int rowIdx = 0; rowIdx < (unsigned int)numRows; rowIdx++)
		{
			for (unsigned int colIdx = 0; colIdx < (unsigned int)numCols; colIdx++)
			{
				fscanf_s(fp, "%f,", &curSensitivity);
				matSensitivity.at<float>(rowIdx, colIdx) = curSensitivity;
			}
			fscanf_s(fp, "\n");
		}

		fclose(fp);
		return true;
	}
	catch (DWORD dwError)
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

		for (unsigned int rowIdx = 0; rowIdx < (unsigned int)numRows; rowIdx++)
		{
			for (unsigned int colIdx = 0; colIdx < (unsigned int)numCols; colIdx++)
			{
				fscanf_s(fp, "%f,", &curSensitivity);
				matDistance.at<float>(rowIdx, colIdx) = curSensitivity;
			}
			fscanf_s(fp, "\n");
		}

		fclose(fp);
		return true;
	}
	catch (DWORD dwError)
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
bool CPSNWhere_Associator3D::CheckVisibility(PSN_Point3D testPoint, unsigned int camIdx, PSN_Point2D *result2DPoint)
{
	PSN_Point2D reprojectedPoint = this->WorldToImage(testPoint, camIdx);
	PSN_Point2D reprojectedTopPoint = this->WorldToImage(PSN_Point3D(testPoint.x, testPoint.y, DEFAULT_HEIGHT), camIdx);
	double halfWidth = (reprojectedTopPoint - reprojectedPoint).norm_L2() / 6;
	if (NULL != result2DPoint) { *result2DPoint = reprojectedPoint; }
	if (reprojectedPoint.x < halfWidth || reprojectedPoint.x >= (double)cCamModel_[camIdx].width() - halfWidth || 
		reprojectedPoint.y < halfWidth || reprojectedPoint.y >= (double)cCamModel_[camIdx].height() - halfWidth)
	{
		return false;
	}
	return true;

	//// http://stackoverflow.com/questions/11716268/point-in-polygon-algorithm
	//bool cross = false;
	//for (int i = 0, j = 3; i < 4; j = i++)
	//{
	//	if (((pFieldOfView[camIdx][i].y >= testPoint.y) != (pFieldOfView[camIdx][j].y >= testPoint.y)) && 
	//		(testPoint.x <= (pFieldOfView[camIdx][j].x - pFieldOfView[camIdx][i].x) * 
	//		(testPoint.y - pFieldOfView[camIdx][i].y) / 
	//		(pFieldOfView[camIdx][j].y - pFieldOfView[camIdx][i].y) + pFieldOfView[camIdx][i].x))
	//	{
	//		cross = !cross;
	//	}
	//}

	//return cross;
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

	if (MAX_HEAD_WIDTH < width3D1 && MAX_HEAD_WIDTH < width3D1) { return false; }
	if (MAX_HEAD_WIDTH < width3D1 && WIDTH_DIFF_RATIO_THRESHOLD * width3D1 > width3D2) { return false; }
	if (MAX_HEAD_WIDTH < width3D1 && WIDTH_DIFF_RATIO_THRESHOLD * width3D2 > width3D1) { return false; }
	return true;
}

/************************************************************************
 Method Name: CheckTrackletConnectivity
 Description: 
	- 
 Input Arguments:
	- 
	- 
 Return Values:
	- 
************************************************************************/
bool CPSNWhere_Associator3D::CheckTrackletConnectivity(PSN_Point3D endPoint, PSN_Point3D startPoint, int timeGap)
{
	if (timeGap > 1) { return true; }
	double norm2 = (endPoint - startPoint).norm_L2();
	return norm2 <= COST_TRACKLET_LINK_MIN_DIST;
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
	resultReconstruction.maxError = 0.0;
	resultReconstruction.costReconstruction = DBL_MAX;
	resultReconstruction.costLink = 0.0;
	resultReconstruction.velocity = PSN_Point3D(0.0, 0.0, 0.0);
	
	if (0 == tracklet2Ds.size()) { return resultReconstruction; }

	resultReconstruction.bIsMeasurement = true;
	resultReconstruction.tracklet2Ds = tracklet2Ds;	
	double fDistance = DBL_MAX;
	double maxError = E_CAL;
	//double maxError = 0;
	double probabilityReconstruction = 0.0;
	PSN_Point2D curPoint(0.0, 0.0);	

	//-------------------------------------------------
	// POINT RECONSTRUCTION
	//-------------------------------------------------
	switch (PSN_DETECTION_TYPE)
	{
	case 1:
		// Full-body
		{						
			std::vector<PSN_Point2D_CamIdx> vecPointInfos;
			for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				stTracklet2D *curTracklet = resultReconstruction.tracklet2Ds.get(camIdx);
				if (NULL == curTracklet) { continue; }

				curPoint = curTracklet->rects.back().reconstructionPoint();				
				vecPointInfos.push_back(PSN_Point2D_CamIdx(curPoint, camIdx));
				maxError += E_DET * matProjectionSensitivity_[camIdx].at<float>((int)curPoint.y, (int)curPoint.x);
				//maxError = std::max(maxError, E_DET * (double)matProjectionSensitivity_[camIdx].at<float>((int)curPoint.y, (int)curPoint.x));

				// 3D position
				resultReconstruction.rawPoints.push_back(curTracklet->currentLocation3D);

				//// sensitivity
				//resultReconstruction.maxError += matProjectionSensitivity_[camIdx].at<float>((int)curPoint.y, (int)curPoint.x);
			}
			//if (!CONSIDER_SENSITIVITY) { maxError = (double)MAX_BODY_WIDHT / 2.0; }
			//else { maxError += E_CAL; }
			resultReconstruction.maxError = maxError;
			fDistance = this->NViewGroundingPointReconstruction(vecPointInfos, resultReconstruction.point);
		}		
		break;
	default:
		// Head
		{
			std::vector<PSN_Line> vecBackprojectionLines;
			for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				stTracklet2D *curTracklet = resultReconstruction.tracklet2Ds.get(camIdx);
				if (NULL == curTracklet) { continue; }
								
				curPoint = curTracklet->rects.back().reconstructionPoint(); 				
				PSN_Line curLine = curTracklet->backprojectionLines.back();

				vecBackprojectionLines.push_back(curLine);
				//maxError += E_DET * matProjectionSensitivity_[camIdx].at<float>((int)curPoint.y, (int)curPoint.x);
				maxError = std::max(maxError, E_DET * (double)matProjectionSensitivity_[camIdx].at<float>((int)curPoint.y, (int)curPoint.x));

				// 3D position
				resultReconstruction.rawPoints.push_back(curTracklet->currentLocation3D);

				//// sensitivity
				//resultReconstruction.maxError += matProjectionSensitivity_[camIdx].at<float>((int)curPoint.y, (int)curPoint.x);
			}
			if (!CONSIDER_SENSITIVITY) { maxError = (double)MAX_BODY_WIDHT / 2.0; }
			else { maxError += E_CAL; }
			resultReconstruction.maxError = maxError;
			fDistance = this->NViewPointReconstruction(vecBackprojectionLines, resultReconstruction.point);
		}
		break;
	}
	resultReconstruction.smoothedPoint = resultReconstruction.point;

	if (2 > tracklet2Ds.size())
	{
		probabilityReconstruction = 0.5;
	}
	else
	{
		if (fDistance > maxError) { return resultReconstruction; }
		probabilityReconstruction = 1 == resultReconstruction.tracklet2Ds.size()? 0.5 : 0.5 * psn::erfc(4.0 * fDistance / maxError - 2.0);
	}

	//-------------------------------------------------
	// DETECTION PROBABILITY
	//-------------------------------------------------
	double fDetectionProbabilityRatio = 1.0;
	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (!this->CheckVisibility(resultReconstruction.point, camIdx)) { continue; }
		if (NULL == resultReconstruction.tracklet2Ds.get(camIdx))
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
	resultReconstruction.costSmoothedPoint = resultReconstruction.costReconstruction;

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
	for (unsigned int lineIdx = 0; lineIdx < numLines; lineIdx++)
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
	for (unsigned int lineIdx = 0; lineIdx < numLines; lineIdx++)
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
	for (unsigned int pointIdx = 0; pointIdx < numPoints; pointIdx++)
	{
		PSN_Point2D curPoint = vecPointInfos[pointIdx].first;
		unsigned int curCamIdx = vecPointInfos[pointIdx].second;
		vecReconstructedPoints.push_back(this->ImageToWorld(curPoint, 0, curCamIdx));
		if (!CONSIDER_SENSITIVITY) 
		{
			outputPoint += vecReconstructedPoints.back();
			continue; 
		}

		int sensitivityX = std::min(std::max(0, (int)curPoint.x), matProjectionSensitivity_[curCamIdx].cols);
		int sensitivityY = std::min(std::max(0, (int)curPoint.y), matProjectionSensitivity_[curCamIdx].rows);
		double curSensitivity = matProjectionSensitivity_[curCamIdx].at<float>(sensitivityY, sensitivityX);
		double invSensitivity = 1.0 / curSensitivity;
		
		vecInvSensitivity.push_back(invSensitivity);
		outputPoint += vecReconstructedPoints.back() * invSensitivity;
		sumInvSensitivity += invSensitivity;
	}
	if (CONSIDER_SENSITIVITY) 
		outputPoint /= sumInvSensitivity;
	else
		outputPoint /= (double)numPoints;
	
	if (2 > numPoints) 
	{ 
		if (CONSIDER_SENSITIVITY)
			return (double)MAX_TRACKLET_SENSITIVITY_ERROR; 
		else
			return (double)MAX_BODY_WIDHT / 2.0;
	}
	for (unsigned int pointIdx = 0; pointIdx < numPoints; pointIdx++)
	{
		if (CONSIDER_SENSITIVITY)
			resultError += (1.0/(double)numPoints) * vecInvSensitivity[pointIdx] * (outputPoint - vecReconstructedPoints[pointIdx]).norm_L2();
		else
			resultError += (1.0/(double)numPoints) * (outputPoint - vecReconstructedPoints[pointIdx]).norm_L2();
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
 Method Name: GetDistanceFromBoundary
 Description: 
	- 
 Input Arguments:
	- 
	- 
 Return Values:
	- 
************************************************************************/
double CPSNWhere_Associator3D::GetDistanceFromBoundary(PSN_Point3D point)
{
	PSN_Point2D curPoint;
	double maxDistance = -100.0, curDistance = 0.0;
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (!CheckVisibility(point, camIdx, &curPoint)) { continue; }
		curDistance = matDistanceFromBoundary_[camIdx].at<float>((int)curPoint.y, (int)curPoint.x);
		if (maxDistance < curDistance) { maxDistance = curDistance; }
	}
	return maxDistance;
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
	if (NUM_CAM != curTrack2DResult.size()) { return; }

	/////////////////////////////////////////////////////////////////////
	// 2D TRACKLET UPDATE
	/////////////////////////////////////////////////////////////////////
	// find update infos (real updating will be done after this loop) and generate new tracklets
	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		//this->m_matDetectionMap[camIdx] = cv::Mat::zeros(this->m_matDetectionMap[camIdx].rows, this->m_matDetectionMap[camIdx].cols, CV_64FC1);
		vecTracklet2DSet_[camIdx].newMeasurements.clear();
		if (frameIdx != curTrack2DResult[camIdx].frameIdx) { continue; }
		if (camIdx != curTrack2DResult[camIdx].camID) { continue; }

		unsigned int numObject = (unsigned int)curTrack2DResult[camIdx].object2DInfos.size();
		unsigned int numTracklet = (unsigned int)vecTracklet2DSet_[camIdx].activeTracklets.size();
		std::vector<stObject2DInfo*> tracklet2DUpdateInfos(numTracklet, NULL);

		//-------------------------------------------------
		// MATCHING AND GENERATING NEW 2D TRACKLET
		//-------------------------------------------------
		for (unsigned int objectIdx = 0; objectIdx < numObject; objectIdx++)
		{
			stObject2DInfo *curObject = &curTrack2DResult[camIdx].object2DInfos[objectIdx];

			// detection size crop
			curObject->box = curObject->box.cropWithSize(cCamModel_[camIdx].width(), cCamModel_[camIdx].height());

			// set detection map
			//this->m_matDetectionMap[camIdx](curObject->box.cv()) = 1.0;

			// find appropriate 2D tracklet
			bool bNewTracklet2D = true; 
			for (unsigned int tracklet2DIdx = 0; tracklet2DIdx < numTracklet; tracklet2DIdx++)
			{
				if (curObject->id == vecTracklet2DSet_[camIdx].activeTracklets[tracklet2DIdx]->id)
				{
					bNewTracklet2D = false;
					tracklet2DUpdateInfos[tracklet2DIdx] = curObject;
					break;
				}
			}
			if (!bNewTracklet2D) { continue; }

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

			// appearance
			cv::Rect cropRect = curObject->box.cv();
			if (cropRect.x < 0) { cropRect.x = 0; }
			if (cropRect.y < 0) { cropRect.y = 0; }
			if (cropRect.x + cropRect.width > ptMatCurrentFrames_[camIdx]->cols) { cropRect.width = ptMatCurrentFrames_[camIdx]->cols - cropRect.x; }
			if (cropRect.y + cropRect.height > ptMatCurrentFrames_[camIdx]->rows) { cropRect.height = ptMatCurrentFrames_[camIdx]->rows - cropRect.y; }
			cv::Mat patch = (*ptMatCurrentFrames_[camIdx])(cropRect);
			newTracklet.RGBFeatureHead = GetRGBFeature(&patch, NUM_BINS_RGB_HISTOGRAM);
			newTracklet.RGBFeatureTail = newTracklet.RGBFeatureHead.clone();

			// location in 3D
			newTracklet.currentLocation3D = this->ImageToWorld(curObject->box.bottomCenter(), 0.0, camIdx);

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
		for (unsigned int objInfoIdx = 0; objInfoIdx < numTracklet; objInfoIdx++)
		{
			if (NULL == tracklet2DUpdateInfos[objInfoIdx])
			{
				if (!(*trackletIter)->bActivated)
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

			// appearance
			cv::Rect cropRect = curTracklet->rects.back().cv();
			if (cropRect.x < 0) { cropRect.x = 0; }
			if (cropRect.y < 0) { cropRect.y = 0; }
			if (cropRect.x + cropRect.width > ptMatCurrentFrames_[camIdx]->cols) { cropRect.width = ptMatCurrentFrames_[camIdx]->cols - cropRect.x; }
			if (cropRect.y + cropRect.height > ptMatCurrentFrames_[camIdx]->rows) { cropRect.height = ptMatCurrentFrames_[camIdx]->rows - cropRect.y; }
			cv::Mat patch = (*ptMatCurrentFrames_[camIdx])(cropRect);
			curTracklet->RGBFeatureTail = GetRGBFeature(&patch, NUM_BINS_RGB_HISTOGRAM);

			// location in 3D
			curTracklet->currentLocation3D = this->ImageToWorld(curObject->box.bottomCenter(), 0.0, camIdx);

			// association informations
			for (unsigned int subloopCamIdx = 0; subloopCamIdx < NUM_CAM; subloopCamIdx++)
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
	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (!bReceiveNewMeasurement_) { break; }

		for (std::deque<stTracklet2D*>::iterator trackletIter = vecTracklet2DSet_[camIdx].activeTracklets.begin();
			trackletIter != vecTracklet2DSet_[camIdx].activeTracklets.end();
			trackletIter++)
		{		
			PSN_Line backProjectionLine1 = (*trackletIter)->backprojectionLines.back();
			for (unsigned int subloopCamIdx = 0; subloopCamIdx < NUM_CAM; subloopCamIdx++)
			{
				unsigned int numNewMeasurements = (unsigned int)vecTracklet2DSet_[subloopCamIdx].newMeasurements.size();
				(*trackletIter)->bAssociableNewMeasurement[subloopCamIdx].resize(numNewMeasurements, false);

				if (camIdx == subloopCamIdx) { continue; }
				for (unsigned int measurementIdx = 0; measurementIdx < numNewMeasurements; measurementIdx++)
				{
					stTracklet2D *curMeasurement = vecTracklet2DSet_[subloopCamIdx].newMeasurements[measurementIdx];
					PSN_Line backProjectionLine2 = curMeasurement->backprojectionLines.back();

					PSN_Point3D reconstructedPoint;
					std::vector<PSN_Line> vecBackProjectionLines;
					vecBackProjectionLines.push_back(backProjectionLine1);
					vecBackProjectionLines.push_back(backProjectionLine2);

					double fDistance = this->NViewPointReconstruction(vecBackProjectionLines, reconstructedPoint);
					//double fDistance = this->StereoTrackletReconstruction(*trackletIter, curMeasurement, reconstructedPoints);

					if (MAX_TRACKLET_DISTANCE >= fDistance)
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
	if (camIdx >= NUM_CAM)
	{
#ifdef PSN_DEBUG_MODE
		//combination.print();
#endif
		combinationQueue.push_back(combination);
		return;
	}

	if (NULL != combination.get(camIdx))
	{
		std::vector<bool> vecNewBAssociationMap[NUM_CAM];
		for (unsigned int subloopCamIdx = 0; subloopCamIdx < NUM_CAM; subloopCamIdx++)
		{
			vecNewBAssociationMap[subloopCamIdx] = vecBAssociationMap[subloopCamIdx];
			if (subloopCamIdx <= camIdx) { continue; }
			for (unsigned int mapIdx = 0; mapIdx < vecNewBAssociationMap[subloopCamIdx].size(); mapIdx++)
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

	for (unsigned int measurementIdx = 0; measurementIdx < vecTracklet2DSet_[camIdx].newMeasurements.size(); measurementIdx++)
	{
		if (!vecBAssociationMap[camIdx][measurementIdx]) { continue; }

		combination.set(camIdx, vecTracklet2DSet_[camIdx].newMeasurements[measurementIdx]);
		// AND operation of map
		std::vector<bool> vecNewBAssociationMap[NUM_CAM];
		for (unsigned int subloopCamIdx = 0; subloopCamIdx < NUM_CAM; subloopCamIdx++)
		{
			vecNewBAssociationMap[subloopCamIdx] = vecBAssociationMap[subloopCamIdx];
			if (subloopCamIdx <= camIdx) { continue; }
			for (unsigned int mapIdx = 0; mapIdx < vecNewBAssociationMap[subloopCamIdx].size(); mapIdx++)
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
	for (std::deque<Track3D*>::iterator trackIter = queueActiveTrack_.begin();
		trackIter != queueActiveTrack_.end();
		trackIter++)
	{
		if (!(*trackIter)->bValid) { continue; }
		Track3D *curTrack = *trackIter;

		// updating current tracklet information
		for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if (NULL == curTrack->curTracklet2Ds.get(camIdx)) { continue; }
			if (!curTrack->curTracklet2Ds.get(camIdx)->bActivated)
			{
				if (MIN_TRACKLET_LENGTH > curTrack->curTracklet2Ds.get(camIdx)->duration)
				{
					// invalidate track with 2D tracklet which has duration 1
					TrackTree::SetValidityFlagInTrackBranch(curTrack, false);
					break;
				}
				curTrack->curTracklet2Ds.set(camIdx, NULL);
			}
			else
			{
				// tracklet location in 3D
				curTrack->trackletLastLocation3D[camIdx] = curTrack->curTracklet2Ds.get(camIdx)->currentLocation3D;

				// RGB update
				curTrack->lastRGBFeature[camIdx].release();
				curTrack->lastRGBFeature[camIdx] = curTrack->curTracklet2Ds.get(camIdx)->RGBFeatureTail.clone();
				curTrack->timeTrackletEnded[camIdx] = nCurrentFrameIdx_;
			}
		}
		if (!curTrack->bValid) { continue; }	

		// activation check and expiration
		if (0 == curTrack->curTracklet2Ds.size())
		{ 		
			// de-activating
			curTrack->bActive = false;

			// cost update (with exit cost)
			std::vector<PSN_Point3D> pointsIn3D;
			stReconstruction lastReconstruction = curTrack->reconstructions.back();
			for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				if (NULL == lastReconstruction.tracklet2Ds.get(camIdx)) { continue; }
				pointsIn3D.push_back(curTrack->trackletLastLocation3D[camIdx]);
			}
			curTrack->costExit = std::min(COST_EX_MAX, -std::log(this->ComputeExitProbability(pointsIn3D)));
			curTrack->costTotal = GetCost(curTrack);

			// move to time jump track list
			queuePausedTrack_.push_back(curTrack);
			continue;
		}

		// cost update (not essential, for just debuging)
		curTrack->costTotal = GetCost(curTrack);
		
		// point reconstruction and cost update
		stReconstruction newReconstruction = this->PointReconstruction(curTrack->curTracklet2Ds);
		newReconstruction.costSmoothedPoint = newReconstruction.costReconstruction;;
		newReconstruction.smoothedPoint = newReconstruction.point;

		//double curLinkProbability = ComputeLinkProbability(curTrack->reconstructions.back().point + curTrack->reconstructions.back().velocity, curReconstruction.point, 1);
		double curLinkProbability = ComputeLinkProbability(curTrack->reconstructions.back().point, newReconstruction.point);
		if (DBL_MAX == newReconstruction.costReconstruction || MIN_LINKING_PROBABILITY > curLinkProbability)
		{
			// invalidate track
			curTrack->bValid = false;			
			//this->SetValidityFlagInTrackBranch(curTrack, false);
			continue;
		}
		newReconstruction.costLink = -log(curLinkProbability);
		newReconstruction.velocity = newReconstruction.point - curTrack->reconstructions.back().smoothedPoint;

		curTrack->reconstructions.push_back(newReconstruction);
		curTrack->costReconstruction += newReconstruction.costSmoothedPoint;
		curTrack->costLink += newReconstruction.costLink;
		curTrack->duration++;
		curTrack->timeEnd++;

		// smoothing		
		int updateStartPos = curTrack->smoother.Insert(newReconstruction.point);
		bool trackValidity = true;
		double smoothedProbability = 0.0;
		for (int pos = updateStartPos; pos < curTrack->smoother.size(); pos++)
		{
			stReconstruction *curReconstruction = &curTrack->reconstructions[pos];			
			if (MIN_SMOOTHING_LENGTH <= curTrack->duration)
			{
				curReconstruction->smoothedPoint = curTrack->smoother.GetResult(pos);
				//curTrack->smoothedTrajectory[pos] = curReconstruction->smoothedPoint;

				// update reconstruction cost
				curTrack->costReconstruction -= curReconstruction->costSmoothedPoint;
				smoothedProbability = ComputeReconstructionProbability(curReconstruction->smoothedPoint, &curReconstruction->rawPoints, &curReconstruction->tracklet2Ds, curReconstruction->maxError);
				if (0.0 == smoothedProbability)
				{
					trackValidity = false;
					break;
				}
				curReconstruction->costSmoothedPoint = -log(smoothedProbability);
				curTrack->costReconstruction += curReconstruction->costSmoothedPoint;

				if (0 == pos) { continue; }
				// update link cost
				curTrack->costLink -= curReconstruction->costLink;

				// TESTING
				smoothedProbability = ComputeLinkProbability(curTrack->reconstructions[pos-1].smoothedPoint, curReconstruction->smoothedPoint);
				//smoothedProbability = ComputeLinkProbability(curTrack->reconstructions[pos-1].smoothedPoint + curTrack->reconstructions[pos-1].velocity, curReconstruction->smoothedPoint);

				if (0.0 == smoothedProbability)
				{
					trackValidity = false;
					break;
				}
				curReconstruction->costLink = -log(smoothedProbability);
				curTrack->costLink += curReconstruction->costLink;
			}
			else
			{
				curReconstruction->smoothedPoint = curReconstruction->point;
				if (0 == pos) { continue; }
			}

			// update velocity
			curReconstruction->velocity = curReconstruction->smoothedPoint - curTrack->reconstructions[pos-1].smoothedPoint;
			if (MIN_MOVING_SPEED > curReconstruction->velocity.norm_L2()) { curReconstruction->velocity = PSN_Point3D(0.0, 0.0, 0.0); }
		}

		// increase iterator
		queueNewTracks.push_back(curTrack);
	}
	queueActiveTrack_ = queueNewTracks;
	queueNewTracks.clear();


	//---------------------------------------------------------
	// UPDATE DE-ACTIVATED TRACKS FOR TEMPORAL BRANCHING
	//---------------------------------------------------------
	std::deque<Track3D*> queueTerminatedTracks;
	for (std::deque<Track3D*>::iterator trackIter = queuePausedTrack_.begin();
		trackIter != queuePausedTrack_.end();
		trackIter++)
	{
		if (!(*trackIter)->bValid) { continue; }

		// handling expired track
		if ((*trackIter)->timeEnd + MAX_TIME_JUMP < nCurrentFrameIdx_)
		{
			// remove track instance for memory efficiency
			if (0.0 <= (*trackIter)->costTotal)
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

		// insert dummy reconstruction
		stReconstruction dummyReconstruction = this->PointReconstruction((*trackIter)->curTracklet2Ds);
		dummyReconstruction.bIsMeasurement = false;
		dummyReconstruction.point = (*trackIter)->reconstructions.back().smoothedPoint + (*trackIter)->reconstructions.back().velocity;
		dummyReconstruction.smoothedPoint = dummyReconstruction.point;
		dummyReconstruction.costLink = 0.0;
		dummyReconstruction.costReconstruction = 0.0;
		dummyReconstruction.costSmoothedPoint = 0.0;
		dummyReconstruction.maxError = 0.0;
		dummyReconstruction.velocity = (*trackIter)->reconstructions.back().velocity;
		(*trackIter)->reconstructions.push_back(dummyReconstruction);
						
		queueNewTracks.push_back(*trackIter);
	}
	queuePausedTrack_ = queueNewTracks;
	queueNewTracks.clear();

	// print out
#ifdef PSN_PRINT_LOG_
	//this->PrintTracks(queueTerminatedTracks, strTrackLogFileName_, true);
#endif
	queueTerminatedTracks.clear();

	//---------------------------------------------------------
	// MANAGE TRACKS IN PROCESSING WINDOW
	//---------------------------------------------------------
	for (std::deque<Track3D*>::iterator trackIter = queueTracksInWindow_.begin();
		trackIter != queueTracksInWindow_.end();
		trackIter++)
	{
		if (!(*trackIter)->bValid
			|| (*trackIter)->timeEnd + PROC_WINDOW_SIZE <= nCurrentFrameIdx_)
		{ continue; }

		queueNewTracks.push_back(*trackIter);
		//if ((*trackIter)->tree->maxGTProb < (*trackIter)->GTProb)
		//{
		//	(*trackIter)->tree->maxGTProb = (*trackIter)->GTProb;
		//}
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
		double GTPSum = 0.0;
		std::deque<Track3D*> queueUpdated; // copy is faster than delete
		for (std::deque<Track3D*>::iterator trackIter = (*treeIter)->tracks.begin();
			trackIter != (*treeIter)->tracks.end();
			trackIter++)
		{
			GTPSum += (*trackIter)->GTProb;
			// reset fields for optimization
			(*trackIter)->BranchGTProb = 0.0;
			(*trackIter)->GTProb = 0.0;
			(*trackIter)->bCurrentBestSolution = false;

			//if (!(*trackIter)->bValid && (*trackIter)->timeGeneration + PROC_WINDOW_SIZE > nCurrentFrameIdx_) { continue; }
			if (!(*trackIter)->bValid) { continue; }
			queueUpdated.push_back(*trackIter);
		}
		(*treeIter)->tracks = queueUpdated;
		//if (0 == (*treeIter)->tracks.size() || (0 == GTPSum && (*treeIter)->timeGeneration + NUM_FRAME_FOR_CONFIRMATION <= nCurrentFrameIdx_))
		if (0 == (*treeIter)->tracks.size())
		{
			(*treeIter)->bValid = false;
			continue;
		}

		//// update 2D tracklet info
		//(*treeIter)->numMeasurements = 0;
		//for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		//{
		//	if (0 == (*treeIter)->tracklet2Ds[camIdx].size()) { continue; }

		//	std::deque<stTracklet2DInfo> queueUpdatedTrackletInfo;
		//	for (std::deque<stTracklet2DInfo>::iterator infoIter = (*treeIter)->tracklet2Ds[camIdx].begin();
		//		infoIter != (*treeIter)->tracklet2Ds[camIdx].end();
		//		infoIter++)
		//	{
		//		if ((*infoIter).tracklet2D->timeStart + nNumFramesForProc_ < nCurrentFrameIdx_)
		//		{ continue;	}

		//		std::deque<Track3D*> queueNewTracks;
		//		for (std::deque<Track3D*>::iterator trackIter = (*infoIter).queueRelatedTracks.begin();
		//			trackIter != (*infoIter).queueRelatedTracks.end();
		//			trackIter++)
		//		{
		//			if ((*trackIter)->bValid) { queueNewTracks.push_back(*trackIter); }
		//		}

		//		if (0 == queueNewTracks.size()) { continue; }

		//		(*infoIter).queueRelatedTracks = queueNewTracks;
		//		queueUpdatedTrackletInfo.push_back(*infoIter);
		//	}
		//	(*treeIter)->numMeasurements += (unsigned int)queueUpdatedTrackletInfo.size();
		//	(*treeIter)->tracklet2Ds[camIdx] = queueUpdatedTrackletInfo;
		//}
		queueNewActiveTrees.push_back(*treeIter);
	}	
	queuePtActiveTrees_ = queueNewActiveTrees;
	queueNewActiveTrees.clear();

	// update unconfirmed trees
	std::deque<TrackTree*> queueNewUnconfirmedTrees;
	for (std::deque<TrackTree*>::iterator treeIter = queuePtUnconfirmedTrees_.begin();
		treeIter != queuePtUnconfirmedTrees_.end();
		treeIter++)
	{
		if (!(*treeIter)->bValid || (*treeIter)->timeGeneration + NUM_FRAME_FOR_CONFIRMATION <= nCurrentFrameIdx_)
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
		if (trackIter->bValid)
		{	
			// clear new flag
			trackIter->bNewTrack = false;
			trackIter++;
			continue;
		}		
		
		// delete the instance only when the tree is invalidated because of preserving the data structure
		if (trackIter->tree->bValid)
		{
			trackIter++;
			continue;				
		}
		trackIter = listTrack3D_.erase(trackIter);
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
		newTrack.costRGB = 0.0;
		std::vector<PSN_Point3D> pointsIn3D;
		for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			newTrack.timeTrackletEnded[camIdx] = 0;
			if(NULL == newTrack.curTracklet2Ds.get(camIdx)){ continue; }
			newTrack.tracklet2DIDs[camIdx].push_back(newTrack.curTracklet2Ds.get(camIdx)->id);
			pointsIn3D.push_back(newTrack.curTracklet2Ds.get(camIdx)->currentLocation3D);

			// tracklet location in 3D
			newTrack.trackletLastLocation3D[camIdx] = newTrack.curTracklet2Ds.get(camIdx)->currentLocation3D;
			
			// appearance
			newTrack.lastRGBFeature[camIdx] = newTrack.curTracklet2Ds.get(camIdx)->RGBFeatureTail.clone();
			newTrack.timeTrackletEnded[camIdx] = nCurrentFrameIdx_;
		}

		// initiation cost
		//newTrack.costEnter = bInitiationPenaltyFree_? -log(P_EN_MAX) : std::min(COST_EN_MAX, -std::log(this->ComputeEnterProbability(pointsIn3D)));
		newTrack.costEnter = bInitiationPenaltyFree_? 0 : std::min(COST_EN_MAX, -std::log(this->ComputeEnterProbability(pointsIn3D)));
		pointsIn3D.clear();

		// point reconstruction
		stReconstruction curReconstruction = this->PointReconstruction(newTrack.curTracklet2Ds);
		if (DBL_MAX == curReconstruction.costReconstruction) { continue; }
		newTrack.costReconstruction = curReconstruction.costReconstruction;
		newTrack.reconstructions.push_back(curReconstruction);
		
		// smoothing
		newTrack.smoother.SetQsets(&precomputedQsets);
		newTrack.smoother.Insert(curReconstruction.point);
		//newTrack.smoothedTrajectory.push_back(newTrack.smoother.GetResult(0));
		
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

	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		// TODO: 브랜치 뜬 track을, tracklet 안에서 visual feature distance가 가장 작은 것부터 sorting, 점증적으로 cost 주기
	}

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

		if (0 == queueBranches.size()) { continue; }
		if (1 == queueBranches.size() && queueBranches[0] == curCombination) { continue; }

		//---------------------------------------------------------
		// BRANCHING
		//---------------------------------------------------------
		for (std::deque<CTrackletCombination>::iterator branchIter = queueBranches.begin();
			branchIter != queueBranches.end();
			branchIter++)
		{
			if (*branchIter == curCombination){ continue; }
			Track3D *curTrack = *trackIter;

			// generate a new track with branching combination
			Track3D newTrack;
			newTrack.id = nNewTrackID_;
			newTrack.curTracklet2Ds = *branchIter;	
			newTrack.bActive = true;
			newTrack.bValid = true;
			newTrack.tree = curTrack->tree;
			newTrack.parentTrack = curTrack;
			newTrack.childrenTrack.clear();
			newTrack.timeStart = curTrack->timeStart;
			newTrack.timeEnd = curTrack->timeEnd;
			newTrack.timeGeneration = nCurrentFrameIdx_;
			newTrack.duration = curTrack->duration;	
			newTrack.bWasBestSolution = true;
			newTrack.bCurrentBestSolution = false;
			newTrack.bNewTrack = true;
			newTrack.GTProb = curTrack->GTProb;
			newTrack.costEnter = curTrack->costEnter;
			newTrack.costReconstruction = curTrack->costReconstruction;
			newTrack.costLink = curTrack->costLink;
			newTrack.costRGB = curTrack->costRGB;
			newTrack.costExit = 0.0;
			newTrack.costTotal = 0.0;

			// reconstruction
			stReconstruction newReconstruction = this->PointReconstruction(newTrack.curTracklet2Ds);
			if (DBL_MAX == newReconstruction.costReconstruction) { continue; }

			stReconstruction oldReconstruction = curTrack->reconstructions.back();
			stReconstruction preReconstruction = curTrack->reconstructions[(*trackIter)->reconstructions.size()-2];
			//double curLinkProbability = ComputeLinkProbability(preReconstruction.point + preReconstruction.velocity, curReconstruction.point, 1);
			double curLinkProbability = ComputeLinkProbability(preReconstruction.point, newReconstruction.point);
			if (MIN_LINKING_PROBABILITY > curLinkProbability) {	continue; }

			newTrack.reconstructions = curTrack->reconstructions;
			newTrack.smoother = curTrack->smoother;

			newTrack.reconstructions.pop_back();			
			newTrack.reconstructions.push_back(newReconstruction);			
			newTrack.costReconstruction += newReconstruction.costReconstruction - oldReconstruction.costReconstruction;
			newTrack.costLink += -log(curLinkProbability) - oldReconstruction.costLink;

			// smoothing			
			newTrack.smoother.PopBack();
			int updateStartPos = newTrack.smoother.Insert(newReconstruction.point);
			bool trackValidity = true;
			double smoothedProbability;
			for (int pos = updateStartPos; pos < newTrack.smoother.size(); pos++)
			{
				stReconstruction *curReconstruction = &newTrack.reconstructions[pos];
				if (MIN_SMOOTHING_LENGTH <= curTrack->duration)
				{
					curReconstruction->smoothedPoint = newTrack.smoother.GetResult(pos);
					//newTrack.smoothedTrajectory[pos] = curReconstruction->smoothedPoint;

					// update reconstruction cost
					newTrack.costReconstruction -= curReconstruction->costSmoothedPoint;
					smoothedProbability = ComputeReconstructionProbability(curReconstruction->smoothedPoint, &curReconstruction->rawPoints, &curReconstruction->tracklet2Ds,  curReconstruction->maxError);
					if (0.0 == smoothedProbability)
					{
						trackValidity = false;
						break;
					}
					curReconstruction->costSmoothedPoint = -log(smoothedProbability);
					newTrack.costReconstruction += curReconstruction->costSmoothedPoint;

					if (0 == pos) { continue; }
					// update link cost
					newTrack.costLink -= curReconstruction->costLink;

					// TESTING
					smoothedProbability = ComputeLinkProbability(newTrack.reconstructions[pos-1].smoothedPoint, curReconstruction->smoothedPoint);
					//smoothedProbability = ComputeLinkProbability(newTrack.reconstructions[pos-1].smoothedPoint + newTrack.reconstructions[pos-1].velocity, curReconstruction->smoothedPoint);

					if (0.0 == smoothedProbability)
					{
						trackValidity = false;
						break;
					}
					curReconstruction->costLink = -log(smoothedProbability);
					newTrack.costLink += curReconstruction->costLink;
				}
				else
				{
					curReconstruction->smoothedPoint = curReconstruction->point;
					if (0 == pos) { continue; }
				}
				// update velocity
				curReconstruction->velocity = curReconstruction->smoothedPoint - newTrack.reconstructions[pos-1].smoothedPoint;
				if (MIN_MOVING_SPEED > curReconstruction->velocity.norm_L2()) { curReconstruction->velocity = PSN_Point3D(0.0, 0.0, 0.0); }
			}
			if (!trackValidity) { continue; }

			// DEBUG			
			if (9359 == curTrack->id)
			{
				int a = 0;
			}

			// copy 2D tracklet history and proecssig for clustering + appearance
			newTrack.costRGB = 0.0;
			std::deque<stTracklet2D*> queueNewlyInsertedTracklet2D;
			for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				// location in 3D
				newTrack.trackletLastLocation3D[camIdx] = curTrack->trackletLastLocation3D[camIdx];

				// appearance
				newTrack.lastRGBFeature[camIdx] = curTrack->lastRGBFeature[camIdx].clone();
				newTrack.timeTrackletEnded[camIdx] = curTrack->timeTrackletEnded[camIdx];

				// tracklet info
				newTrack.tracklet2DIDs[camIdx] = curTrack->tracklet2DIDs[camIdx];
				if (NULL == newTrack.curTracklet2Ds.get(camIdx)) { continue; }
				if (0 == newTrack.tracklet2DIDs[camIdx].size() || newTrack.tracklet2DIDs[camIdx].back() != newTrack.curTracklet2Ds.get(camIdx)->id)
				{
					newTrack.tracklet2DIDs[camIdx].push_back(newTrack.curTracklet2Ds.get(camIdx)->id);
					queueNewlyInsertedTracklet2D.push_back(newTrack.curTracklet2Ds.get(camIdx));

					// when a newly inserted tracklet is not the most front tracklet
					if (newTrack.tracklet2DIDs[camIdx].size() > 1)
					{
						int timeGap = nCurrentFrameIdx_ - newTrack.timeTrackletEnded[camIdx];

						// tracklet location
						//newTrack.costLink += ComputeTrackletLinkCost(
						//	newTrack.trackletLastLocation3D[camIdx], 
						//	newTrack.curTracklet2Ds.get(camIdx)->currentLocation3D, 
						//	timeGap);
						trackValidity &= CheckTrackletConnectivity(
							newTrack.trackletLastLocation3D[camIdx], 
							newTrack.curTracklet2Ds.get(camIdx)->currentLocation3D, 
							timeGap);

						// RGB cost
						newTrack.costRGB += ComputeRGBCost(
							&newTrack.lastRGBFeature[camIdx], 
							&newTrack.curTracklet2Ds.get(camIdx)->RGBFeatureHead, 
							timeGap);						
					}
				}
				// appearance
				newTrack.lastRGBFeature[camIdx] = newTrack.curTracklet2Ds.get(camIdx)->RGBFeatureTail.clone();
				newTrack.timeTrackletEnded[camIdx] = nCurrentFrameIdx_;
			}
			if (!trackValidity) { continue; }
			newTrack.costTotal = GetCost(&newTrack);
			
			// generate track instance
			listTrack3D_.push_back(newTrack);
			nNewTrackID_++;
					
			// insert to the track tree and related lists
			Track3D *branchTrack = &listTrack3D_.back();
			curTrack->tree->tracks.push_back(branchTrack);
			curTrack->childrenTrack.push_back(branchTrack);
			queueTracksInWindow_.push_back(branchTrack);

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
	for (std::deque<Track3D*>::iterator trackIter = queuePausedTrack_.begin();
		trackIter != queuePausedTrack_.end();
		trackIter++)
	{
		if (DO_BRANCH_CUT && MAX_TRACK_IN_OPTIMIZATION <= numTemporalBranch) { break; }
		Track3D *curTrack = *trackIter;

		for (std::deque<Track3D*>::iterator seedTrackIter = seedTracks->begin();
			seedTrackIter != seedTracks->end();
			seedTrackIter++)
		{
			Track3D *seedTrack = *seedTrackIter;
			std::deque<stReconstruction> queueSeedReconstruction = (*seedTrackIter)->reconstructions;			

			// link validation
			unsigned int timeGap = seedTrack->timeStart - curTrack->timeEnd;			
			stReconstruction lastMeasurementReconstruction = curTrack->reconstructions[curTrack->duration-1];
			double curLinkProbability = ComputeLinkProbability(lastMeasurementReconstruction.point, seedTrack->reconstructions.front().point, timeGap);
			if (MIN_LINKING_PROBABILITY > curLinkProbability) { continue; }

			// generate a new track with branching combination
			Track3D newTrack;
			newTrack.id = nNewTrackID_;
			newTrack.curTracklet2Ds = seedTrack->curTracklet2Ds;
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
			newTrack.bCurrentBestSolution = false;
			newTrack.bNewTrack = true;
			newTrack.GTProb = curTrack->GTProb;
			newTrack.costEnter = curTrack->costEnter;
			newTrack.costReconstruction = curTrack->costReconstruction;
			newTrack.costLink = curTrack->costLink;
			newTrack.costRGB = curTrack->costRGB;
			newTrack.costExit = 0.0;
			newTrack.costTotal = 0.0;

			// interpolation
			std::vector<stReconstruction> interpolatedReconstructions(timeGap);
			std::vector<PSN_Point3D> interpolatedPoints(timeGap);
			PSN_Point3D delta = (seedTrack->reconstructions.front().point - lastMeasurementReconstruction.point) / (double)timeGap;
			PSN_Point3D lastPosition = lastMeasurementReconstruction.point;
			for (int pos = 0; pos < (int)timeGap - 1; pos++)
			{
				lastPosition += delta;
				interpolatedPoints[pos] = lastPosition;
				interpolatedReconstructions[pos].point = lastPosition;
				interpolatedReconstructions[pos].smoothedPoint = lastPosition;
				interpolatedReconstructions[pos].bIsMeasurement = false;
				interpolatedReconstructions[pos].maxError = 0.0;				
				interpolatedReconstructions[pos].costReconstruction = 0.0;
				interpolatedReconstructions[pos].costSmoothedPoint = 0.0;
				interpolatedReconstructions[pos].costLink = 0.0;
			}
			interpolatedPoints.back() = seedTrack->reconstructions.front().point;
			interpolatedReconstructions.back() = seedTrack->reconstructions.front();

			// smoothing
			newTrack.reconstructions.insert(newTrack.reconstructions.begin(), curTrack->reconstructions.begin(), curTrack->reconstructions.begin() + curTrack->duration);
			newTrack.reconstructions.insert(newTrack.reconstructions.end(), interpolatedReconstructions.begin(), interpolatedReconstructions.end());
			newTrack.smoother = curTrack->smoother;
			int updateStartPos = newTrack.smoother.Insert(interpolatedPoints);
			bool trackValidity = true;
			double smoothedProbability = 0.0;
			for (int pos = updateStartPos; pos < newTrack.smoother.size(); pos++)
			{
				stReconstruction *curReconstruction = &newTrack.reconstructions[pos];
				if (MIN_SMOOTHING_LENGTH <= newTrack.duration)
				{
					curReconstruction->smoothedPoint = newTrack.smoother.GetResult(pos);
					//newTrack.smoothedTrajectory[pos] = curReconstruction->smoothedPoint;

					// update reconstruction cost
					newTrack.costReconstruction -= curReconstruction->costSmoothedPoint;
					smoothedProbability = ComputeReconstructionProbability(curReconstruction->smoothedPoint, &curReconstruction->rawPoints, &curReconstruction->tracklet2Ds, curReconstruction->maxError);
					if (0.0 == smoothedProbability)
					{
						trackValidity = false;
						break;
					}
					curReconstruction->costSmoothedPoint = -log(smoothedProbability);
					newTrack.costReconstruction += curReconstruction->costSmoothedPoint;

					if (0 == pos) { continue; }
					// update link cost
					newTrack.costLink -= curReconstruction->costLink;

					// TESTING
					smoothedProbability = ComputeLinkProbability(newTrack.reconstructions[pos-1].smoothedPoint, curReconstruction->smoothedPoint);
					//smoothedProbability = ComputeLinkProbability(newTrack.reconstructions[pos-1].smoothedPoint + newTrack.reconstructions[pos-1].velocity, curReconstruction->smoothedPoint);

					if (0.0 == smoothedProbability)
					{
						trackValidity = false;
						break;
					}
					curReconstruction->costLink = -log(smoothedProbability);
					newTrack.costLink += curReconstruction->costLink;
				}
				else
				{
					curReconstruction->smoothedPoint = curReconstruction->point;
					if (0 == pos) { continue; }
				}

				// update velocity
				curReconstruction->velocity = curReconstruction->smoothedPoint - newTrack.reconstructions[pos-1].smoothedPoint;
				if (MIN_MOVING_SPEED > curReconstruction->velocity.norm_L2()) { curReconstruction->velocity = PSN_Point3D(0.0, 0.0, 0.0); }
			}
			if (!trackValidity) { continue; }

			// tracklet history
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				// location in 3D
				newTrack.trackletLastLocation3D[camIdx] = (*trackIter)->trackletLastLocation3D[camIdx];

				// appearance
				newTrack.lastRGBFeature[camIdx] = (*trackIter)->lastRGBFeature[camIdx].clone();
				newTrack.timeTrackletEnded[camIdx] = (*trackIter)->timeTrackletEnded[camIdx];

				// tracklet info
				newTrack.tracklet2DIDs[camIdx] = curTrack->tracklet2DIDs[camIdx];
				if (NULL == seedTrack->curTracklet2Ds.get(camIdx)) { continue; }
				if (0 == newTrack.tracklet2DIDs[camIdx].size() || newTrack.tracklet2DIDs[camIdx].back() != seedTrack->tracklet2DIDs[camIdx].back())
				{
					newTrack.tracklet2DIDs[camIdx].push_back(seedTrack->tracklet2DIDs[camIdx].back());

					// when a newly inserted tracklet is not the most front tracklet
					if (newTrack.tracklet2DIDs[camIdx].size() > 1)
					{
						int timeGap = nCurrentFrameIdx_ - newTrack.timeTrackletEnded[camIdx];

						// tracklet location
						//newTrack.costLink += ComputeTrackletLinkCost(
						//	newTrack.trackletLastLocation3D[camIdx], 
						//	newTrack.curTracklet2Ds.get(camIdx)->currentLocation3D, 
						//	timeGap);
						trackValidity &= CheckTrackletConnectivity(
							newTrack.trackletLastLocation3D[camIdx], 
							newTrack.curTracklet2Ds.get(camIdx)->currentLocation3D, 
							timeGap);

						// RGB cost
						newTrack.costRGB += ComputeRGBCost(
							&newTrack.lastRGBFeature[camIdx], 
							&newTrack.curTracklet2Ds.get(camIdx)->RGBFeatureHead, 
							timeGap);
					}
				}
				// appearance
				newTrack.lastRGBFeature[camIdx] = newTrack.curTracklet2Ds.get(camIdx)->RGBFeatureTail.clone();
				newTrack.timeTrackletEnded[camIdx] = nCurrentFrameIdx_;
			}
			if (!trackValidity) { continue; }

			newTrack.costTotal = GetCost(&newTrack);

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
double CPSNWhere_Associator3D::ComputeEnterProbability(std::vector<PSN_Point3D> &points)
{
	double distanceFromBoundary = -100.0;
	for (int pointIdx = 0; pointIdx < points.size(); pointIdx++)
	{
		double curDistance = GetDistanceFromBoundary(points[pointIdx]);
		if (distanceFromBoundary < curDistance) { distanceFromBoundary = curDistance; }
	}
	if (distanceFromBoundary < 0) { return 1.0; }
	return distanceFromBoundary <= BOUNDARY_DISTANCE? 1.0 : P_EN_MAX * exp(-(double)(P_EN_DECAY * (distanceFromBoundary - BOUNDARY_DISTANCE)));
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
double CPSNWhere_Associator3D::ComputeExitProbability(std::vector<PSN_Point3D> &points)
{
	double distanceFromBoundary = -100.0;
	for (int pointIdx = 0; pointIdx < points.size(); pointIdx++)
	{
		double curDistance = GetDistanceFromBoundary(points[pointIdx]);
		if (distanceFromBoundary < curDistance) { distanceFromBoundary = curDistance; }
	}
	if (distanceFromBoundary < 0) { return 1.0; }
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
	return 0.5 * psn::erfc(4.0 * distance / maxDistance - 2.0);
}

/************************************************************************
 Method Name: ComputeRGBCost
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
double CPSNWhere_Associator3D::ComputeTrackletLinkCost(PSN_Point3D preLocation, PSN_Point3D curLocation, int timeGap)
{
	if (timeGap > 1) { return 0.0; }
	double norm2 = (preLocation - curLocation).norm_L2();
	return norm2 > COST_TRACKLET_LINK_MIN_DIST? COST_TRACKLET_LINK_COEF * (norm2 - COST_TRACKLET_LINK_MIN_DIST) : 0.0;
}

/************************************************************************
 Method Name: ComputeReconstructionProbability
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
double CPSNWhere_Associator3D::ComputeReconstructionProbability(PSN_Point3D point, std::vector<PSN_Point3D> *rawPoints, CTrackletCombination *trackletCombination, double maxError)
{
	//-------------------------------------------------
	// RECONSTRUCTION PROBABILITY
	//-------------------------------------------------
	double probabilityReconstruction = 0.5;
	if (1 < rawPoints->size())
	{
		double fDistance = 0.0;
		for (int pointIdx = 0; pointIdx < rawPoints->size(); pointIdx++)
		{
			fDistance += (point - (*rawPoints)[pointIdx]).norm_L2();
		}
		fDistance /= (double)rawPoints->size();

		if (0.0 == maxError) { maxError = CONSIDER_SENSITIVITY? MAX_TRACKLET_SENSITIVITY_ERROR : (double)MAX_BODY_WIDHT / 2.0; }
		if (fDistance > maxError) { return 0.0; }
		probabilityReconstruction = 0.5 * psn::erfc(4.0 * fDistance / maxError - 2.0);
	}

	//-------------------------------------------------
	// DETECTION PROBABILITY
	//-------------------------------------------------
	double fDetectionProbabilityRatio = 1.0;
	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (!this->CheckVisibility(point, camIdx)) { continue; }
		if (NULL == trackletCombination->get(camIdx))
		{
			// false negative
			fDetectionProbabilityRatio *= FN_RATE / (1 - FN_RATE);
			continue;
		}
		// positive
		fDetectionProbabilityRatio *= (1 - FP_RATE) / FP_RATE;
	}	
	return fDetectionProbabilityRatio * probabilityReconstruction / (1.0 - probabilityReconstruction);
}

/************************************************************************
 Method Name: ComputeRGBCost
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
double CPSNWhere_Associator3D::ComputeRGBCost(const cv::Mat *feature1, const cv::Mat *feature2, unsigned int timeGap)
{
	cv::Mat vecDiff = *feature1 - *feature2;
	vecDiff = vecDiff.t() * vecDiff;
	double norm2 = vecDiff.at<double>(0, 0);
	return norm2 > COST_RGB_MIN_DIST? COST_RGB_COEF * std::exp(-COST_RGB_DECAY * (double)(timeGap - 1.0)) * (norm2 - COST_RGB_MIN_DIST) : 0.0;
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
	if (track1->tree->id == track2->tree->id)
	{
		bIncompatible = true;
	}
	else
	{
		for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if (0 == track1->tracklet2DIDs[camIdx].size() || 0 == track2->tracklet2DIDs[camIdx].size())	{ continue;	}
			if (track1->tracklet2DIDs[camIdx][0] > track2->tracklet2DIDs[camIdx][0])
			{
				if (track1->tracklet2DIDs[camIdx][0] > track2->tracklet2DIDs[camIdx].back()) { continue; }
				for (std::deque<unsigned int>::iterator tracklet1Iter = track2->tracklet2DIDs[camIdx].begin();
					tracklet1Iter != track2->tracklet2DIDs[camIdx].end();
					tracklet1Iter++)
				{
					for (std::deque<unsigned int>::iterator tracklet2Iter = track1->tracklet2DIDs[camIdx].begin();
						tracklet2Iter != track1->tracklet2DIDs[camIdx].end();
						tracklet2Iter++)	
					{
						if (*tracklet1Iter != *tracklet2Iter) { continue; }						
						bIncompatible = true;
						break;
					}
					if (bIncompatible) { break;	}
				}
			}
			else if (track1->tracklet2DIDs[camIdx][0] < track2->tracklet2DIDs[camIdx][0])
			{
				if (track1->tracklet2DIDs[camIdx].back() < track2->tracklet2DIDs[camIdx][0]) { continue; }
				for (std::deque<unsigned int>::iterator tracklet1Iter = track1->tracklet2DIDs[camIdx].begin();
					tracklet1Iter != track1->tracklet2DIDs[camIdx].end();
					tracklet1Iter++)
				{
					for (std::deque<unsigned int>::iterator tracklet2Iter = track2->tracklet2DIDs[camIdx].begin();
						tracklet2Iter != track2->tracklet2DIDs[camIdx].end();
						tracklet2Iter++)
					{
						if (*tracklet1Iter != *tracklet2Iter) { continue; }						
						bIncompatible = true;
						break;
					}
					if (bIncompatible) { break;	}
				}
			}
			else
			{
				bIncompatible = true;
				break;
			}
			if (bIncompatible) { break; }
		}
	}

	// check proximity and crossing
	if (!bIncompatible)
	{
		// check overlapping
		if(track1->timeEnd < track2->timeStart || track2->timeEnd < track1->timeStart) { return bIncompatible; }

		unsigned int timeStart = std::max(track1->timeStart, track2->timeStart);
		unsigned int timeEnd = std::min(track1->timeEnd, track2->timeEnd);
		unsigned int track1ReconIdx = timeStart - track1->timeStart;
		unsigned int track2ReconIdx = timeStart - track2->timeStart;
		unsigned int overlapLength = timeEnd - timeStart + 1;
		PSN_Point3D *reconLocation1;
		PSN_Point3D *reconLocation2;
		double distanceBetweenTracks = 0.0;
		for (unsigned int reconIdx = 0; reconIdx < overlapLength; reconIdx++)
		{
			reconLocation1 = &(track1->reconstructions[track1ReconIdx].point);
			reconLocation2 = &(track2->reconstructions[track2ReconIdx].point);
			track1ReconIdx++;
			track2ReconIdx++;
			// proximity
			distanceBetweenTracks = (*reconLocation1 - *reconLocation2).norm_L2();
			if (distanceBetweenTracks > MAX_MOVING_SPEED * 2) { continue; }
			if (distanceBetweenTracks < MIN_TARGET_PROXIMITY) { return true; }
			if (reconIdx < overlapLength - 1)
			{
				// crossing
				if (psn::IsLineSegmentIntersect(PSN_Line(*reconLocation1, track1->reconstructions[track1ReconIdx].point), PSN_Line(*reconLocation2, track2->reconstructions[track2ReconIdx].point)))
				{ return true; }
			}
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
 Method Name: GetRGBFeature
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
cv::Mat CPSNWhere_Associator3D::GetRGBFeature(const cv::Mat *patch, int numBins)
{
	std::vector<cv::Mat> channelImages(3);
	cv::split(*patch, channelImages);
	double numPixels = patch->rows * patch->cols;
	cv::Mat featureB = psn::histogram(channelImages[0], numBins);
	cv::Mat featureG = psn::histogram(channelImages[1], numBins);
	cv::Mat featureR = psn::histogram(channelImages[2], numBins);

	cv::vconcat(featureR, featureG, featureR);
	cv::vconcat(featureR, featureB, featureR);
	featureR = featureR / numPixels;

	return featureR;
}

/************************************************************************
 Method Name: GetCost
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
double CPSNWhere_Associator3D::GetCost(Track3D *track)
{
	track->costReconstruction = 0.0;
	track->costLink = 0.0;
	for (int reconIdx = 0; reconIdx < track->reconstructions.size(); reconIdx++)
	{
		track->costReconstruction += track->reconstructions[reconIdx].costSmoothedPoint;
		track->costLink += track->reconstructions[reconIdx].costLink;
	}
	double resultCost = track->costEnter + track->costReconstruction + track->costLink + track->costRGB + track->costExit;
	return resultCost;
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
	// DEBUG
	nCountTrackInOptimization_ = 0;
	nCountUCTrackInOptimization_ = 0;

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
		PSN_TrackSet newRelatedTracks = (*hypothesisIter).relatedTracks;
		for (PSN_TrackSet::iterator trackIter = (*hypothesisIter).relatedTracks.begin();
			trackIter != (*hypothesisIter).relatedTracks.end();
			trackIter++)
		{		
			for (PSN_TrackSet::iterator childTrackIter = (*trackIter)->childrenTrack.begin();
				childTrackIter != (*trackIter)->childrenTrack.end();
				childTrackIter++)
			{
				if ((*childTrackIter)->bNewTrack) { newRelatedTracks.push_back(*childTrackIter); }
			}
		}
		std::sort(newRelatedTracks.begin(), newRelatedTracks.end(), psnTrackGTPandLLDescend);
		if (newRelatedTracks.size() <= MAX_TRACK_IN_OPTIMIZATION)
		{
			newGlobalHypothesis.relatedTracks = newRelatedTracks;
		}
		else
		{
			newGlobalHypothesis.relatedTracks.insert(newGlobalHypothesis.relatedTracks.end(), newRelatedTracks.begin(), newRelatedTracks.begin() + MAX_TRACK_IN_OPTIMIZATION);
		}
		
		// add new seed tracks
		newGlobalHypothesis.relatedTracks.insert(newGlobalHypothesis.relatedTracks.end(), newSeedTracks->begin(), newSeedTracks->end());
		
		// save hypothesis
		newHypothesesSet.push_back(newGlobalHypothesis);

		// DEBUG
		nCountTrackInOptimization_ += (int)(*hypothesisIter).relatedTracks.size();
		for (PSN_TrackSet::iterator trackIter = (*hypothesisIter).relatedTracks.begin();
			trackIter != (*hypothesisIter).relatedTracks.end();
			trackIter++)
		{
			if ((*trackIter)->tree->timeGeneration + NUM_FRAME_FOR_CONFIRMATION >= nCurrentFrameIdx_) { nCountUCTrackInOptimization_++; }
		}
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
		if (!curTrack->bValid) { continue; }

		// cost	
		curTrack->costTotal = GetCost(curTrack);

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
			if (CheckIncompatibility(curTrack, queueTracksInOptimization[compTrackIdx])) { continue; }
			curGraph->AddEdge(vertexInGraph[compTrackIdx], vertexInGraph.back()); // for MCP
		}

		// initial solution related
		if (NULL == initialselectedTracks) { continue; }
		PSN_TrackSet::iterator findIter = std::find(initialselectedTracks->begin(), initialselectedTracks->end(), curTrack);
		if (initialselectedTracks->end() != findIter)
		{
			vertexInInitialSolution.push_back(curVertex);
		}
	}

	// validate initial solution
	if (0 < vertexInInitialSolution.size())
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

	// DEBUG
	//PrintHypotheses(*ptQueueHypothesis, "data/hypotheses.txt", nCurrentFrameIdx);

	// track tree branch pruning
	int nTimeBranchPruning = (int)nCurrentFrameIdx - (int)N;
	PSN_TrackSet *tracksInBestSolution = &(*ptQueueHypothesis).front().selectedTracks;
	Track3D *brachSeedTrack = NULL;
	for (int trackIdx = 0; trackIdx < tracksInBestSolution->size(); trackIdx++)
	{
		(*tracksInBestSolution)[trackIdx]->bCurrentBestSolution = true;

		// left unconfirmed tracks
		if ((*tracksInBestSolution)[trackIdx]->tree->timeGeneration + NUM_FRAME_FOR_CONFIRMATION > nCurrentFrameIdx){ continue; }

		// pruning
		brachSeedTrack = TrackTree::FindOldestTrackInBranch((*tracksInBestSolution)[trackIdx], nTimeBranchPruning);
		if (NULL == brachSeedTrack->parentTrack) { continue; }
		for (int childIdx = 0; childIdx < brachSeedTrack->parentTrack->childrenTrack.size(); childIdx++)
		{
			if (brachSeedTrack->parentTrack->childrenTrack[childIdx] == brachSeedTrack) { continue; }
			TrackTree::SetValidityFlagInTrackBranch(brachSeedTrack->parentTrack->childrenTrack[childIdx], false);
		}
	}
}

/************************************************************************
 Method Name: Hypothesis_PruningKBest
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
void CPSNWhere_Associator3D::Hypothesis_PruningTrackWithGTP(unsigned int nCurrentFrameIdx, unsigned int nNumMaximumTrack, PSN_TrackSet *tracksInWindow)
{
	int numTrackLeft = 0;
	int numUCTrackLeft = 0;
	
	std::sort(tracksInWindow->begin(), tracksInWindow->end(), psnTrackGTPandLLDescend);	
	for (PSN_TrackSet::iterator trackIter = tracksInWindow->begin();
		trackIter != tracksInWindow->end();
		trackIter++)
	{
		//if((*trackIter)->tree->timeGeneration + NUM_FRAME_FOR_CONFIRMATION > this->m_nCurrentFrameIdx && numUCTrackLeft < MAX_UNCONFIRMED_TRACK) 
		if ((*trackIter)->tree->timeGeneration + NUM_FRAME_FOR_CONFIRMATION > nCurrentFrameIdx) 
		{ 
			numUCTrackLeft++;
			continue; 
		}
		if (numTrackLeft < MAX_TRACK_IN_OPTIMIZATION && (*trackIter)->GTProb > 0) 
		{ 	
			numTrackLeft++;
			continue;
		}
		(*trackIter)->bValid = false;
	}

	// pruning unconfirmed tracks
	for (int treeIdx = 0; treeIdx < queuePtUnconfirmedTrees_.size(); treeIdx++)
	{
		PSN_TrackSet sortedTrackQueue = queuePtUnconfirmedTrees_[treeIdx]->tracks;
		std::sort(sortedTrackQueue.begin(), sortedTrackQueue.end(), psnTrackGTPandLLDescend);
		for (int trackIdx = MAX_TRACK_IN_UNCONFIRMED_TREE; trackIdx < sortedTrackQueue.size(); trackIdx++)
		{
			sortedTrackQueue[trackIdx]->bValid = false;
		}
	}

#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Associator3D](Hypothesis_PruningTrackWithGTP)\n");
	if (MAX_TRACK_IN_OPTIMIZATION <= numTrackLeft) { printf("*** Tracks are truncated!!! ****\n"); }
#endif
}

/************************************************************************
 Method Name: Hypothesis_RefreshHypotheses
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
void CPSNWhere_Associator3D::Hypothesis_RefreshHypotheses(PSN_HypothesisSet &inoutUpdatedHypotheses)
{
	// gathering tracks in unconfirmed track trees
	PSN_TrackSet unconfirmedTracks;
	for (int treeIdx = 0; treeIdx < queuePtUnconfirmedTrees_.size(); treeIdx++)
	{
		for (PSN_TrackSet::iterator trackIter = queuePtUnconfirmedTrees_[treeIdx]->tracks.begin();
			trackIter != queuePtUnconfirmedTrees_[treeIdx]->tracks.end();
			trackIter++)
		{
			if (!(*trackIter)->bValid) { continue; }
			unconfirmedTracks.push_back(*trackIter);
		}
	}

	// update hyptheses
	std::deque<stGlobalHypothesis> existingHypothesis = inoutUpdatedHypotheses;
	inoutUpdatedHypotheses.clear();
	stGlobalHypothesis *curHypotheis = NULL;
	bool bCurHypothesisValid = true;
	for (int hypothesisIdx = 0; hypothesisIdx < existingHypothesis.size(); hypothesisIdx++)
	{
		curHypotheis = &existingHypothesis[hypothesisIdx];

		// check validity
		bCurHypothesisValid = true;
		for (int trackIdx = 0; trackIdx < curHypotheis->selectedTracks.size(); trackIdx++)
		{
			if (curHypotheis->selectedTracks[trackIdx]->bValid) { continue; }
			bCurHypothesisValid = false;
			break;
		}
		if (!bCurHypothesisValid) { continue; }

		// update related track list
		PSN_TrackSet newRelatedTracks = curHypotheis->selectedTracks;
		newRelatedTracks.insert(newRelatedTracks.end(), unconfirmedTracks.begin(), unconfirmedTracks.end());

		// TO-MHT like
		//for (int trackIdx = 0; trackIdx < curHypotheis->relatedTracks.size(); trackIdx++)
		//{
		//	if (!curHypotheis->relatedTracks[trackIdx]->bValid) { continue; }
		//	newRelatedTracks.push_back(curHypotheis->relatedTracks[trackIdx]);
		//}

		curHypotheis->relatedTracks = newRelatedTracks;

		// save valid hypothesis
		inoutUpdatedHypotheses.push_back(*curHypotheis);
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

		// ID
		newObject.id = curTrack->id;
		if (1 == DISPLAY_ID_MODE)
		{
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
				newObject.id = nNewVisualizationID_++;			
				queuePairTreeIDToVisualizationID_.push_back(std::make_pair(curTrack->tree->id, newObject.id));
			}
		}
		
		unsigned numPoint = 0;
		//int deferredLength = std::max((int)curTrack->timeEnd - (int)nFrameIdx, 0); 
		int deferredLength = curTrack->timeEnd - nFrameIdx;
		if (deferredLength > curTrack->reconstructions.size()) { continue; }
		bool bVisible[NUM_CAM];
		for (int camIdx = 0; camIdx < NUM_CAM; camIdx++) { bVisible[camIdx] = false; }
		for (std::deque<stReconstruction>::reverse_iterator pointIter = curTrack->reconstructions.rbegin() + deferredLength;
			pointIter != curTrack->reconstructions.rend();
			pointIter++)
		{
			numPoint++;
			newObject.recentPoints.push_back((*pointIter).smoothedPoint);			
			if (DISP_TRAJECTORY3D_LENGTH < numPoint) { break; }

			for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				//newObject.point3DBox[camIdx] = this->GetHuman3DBox((*pointIter).point, 500, camIdx);
				PSN_Point2D reprejectedPoint(0.0, 0.0);
				if (CheckVisibility((*pointIter).smoothedPoint, camIdx, &reprejectedPoint)) { bVisible[camIdx] = true; }
				newObject.recentPoint2Ds[camIdx].push_back(reprejectedPoint);

				if (1 < numPoint) { continue; }
				if (NULL == curTrack->curTracklet2Ds.get(camIdx))
				{
					PSN_Rect curRect(0.0, 0.0, 0.0, 0.0);
					PSN_Point3D topCenterInWorld((*pointIter).smoothedPoint);
					topCenterInWorld.z = 1700 / CAM_HEIGHT_SCALE[camIdx];
					PSN_Point2D bottomCenterInImage = this->WorldToImage((*pointIter).smoothedPoint, camIdx);
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

		// 2D detection position and visibillity
		for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			stTracklet2D *curTracklet = curTrack->reconstructions.back().tracklet2Ds.get(camIdx);
			PSN_Point3D detectionLocation(0.0, 0.0, 0.0);
			if (NULL != curTracklet) { detectionLocation = this->ImageToWorld(curTracklet->rects.back().bottomCenter(), 0.0, camIdx); }
			newObject.curDetectionPosition.push_back(detectionLocation);

			if (!bVisible[camIdx]) { newObject.recentPoint2Ds[camIdx].clear(); }
		}

		result3D.object3DInfo.push_back(newObject);
	}

	return result3D;
}

/************************************************************************
 Method Name: PrintTracks
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
	if (0 == queueTracks.size()) { return; }
	try
	{
		FILE *fp;
		if (bAppend)
		{
			fopen_s(&fp, strFilePathAndName, "a");
		}
		else
		{
			fopen_s(&fp, strFilePathAndName, "w");
			fprintf_s(fp, "numCamera:%d\n", (int)NUM_CAM);			
			fprintf_s(fp, "numTracks:%d\n", (int)queueTracks.size());			
		}

		for (std::deque<Track3D*>::iterator trackIter = queueTracks.begin();
			trackIter != queueTracks.end();
			trackIter++)
		{
			Track3D *curTrack = *trackIter;
			fprintf_s(fp, "{\n\tid:%d\n\ttreeID:%d\n", (int)curTrack->id, (int)curTrack->tree->id);
			fprintf_s(fp, "\tnumReconstructions:%d\n\ttimeStart:%d\n\ttimeEnd:%d\n\ttimeGeneration:%d\n", (int)curTrack->reconstructions.size(), (int)curTrack->timeStart, (int)curTrack->timeEnd, (int)curTrack->timeGeneration);
			fprintf_s(fp, "\ttrackleIDs:\n\t{\n");
			for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				if (0 == curTrack->tracklet2DIDs[camIdx].size())
				{
					fprintf(fp, "\t\tnumTracklet:0,[]\n");
					continue;
				}
				fprintf_s(fp, "\t\tnumTracklet:%d,[", (int)curTrack->tracklet2DIDs[camIdx].size());				
				for (unsigned int trackletIDIdx = 0; trackletIDIdx < curTrack->tracklet2DIDs[camIdx].size(); trackletIDIdx++)
				{
					fprintf_s(fp, "%d", (int)curTrack->tracklet2DIDs[camIdx][trackletIDIdx]);					
					if (curTrack->tracklet2DIDs[camIdx].size() - 1 != trackletIDIdx)
					{
						fprintf_s(fp, ",");
					}
					else
					{
						fprintf_s(fp, "]\n");
					}
				}
			}
			fprintf(fp, "\t}\n");
			fprintf_s(fp, "\ttotalCost:%e\n", curTrack->costTotal);
			fprintf_s(fp, "\treconstructionCost:%e\n", curTrack->costReconstruction);
			fprintf_s(fp, "\tlinkCost:%e\n", curTrack->costLink);
			fprintf_s(fp, "\tinitCost:%e\n", curTrack->costEnter);
			fprintf_s(fp, "\ttermCost:%e\n", curTrack->costExit);
			fprintf_s(fp, "\tRGBCost:%e\n", curTrack->costRGB);
			fprintf_s(fp, "\treconstructions:\n\t{\n");
			for (std::deque<stReconstruction>::iterator pointIter = curTrack->reconstructions.begin();
				pointIter != curTrack->reconstructions.end();
				pointIter++)
			{
				fprintf_s(fp, "\t\t%d:(%f,%f,%f),%e,%e,%e\n", (int)(*pointIter).bIsMeasurement, (*pointIter).point.x, (*pointIter).point.y, (*pointIter).point.z, (*pointIter).costReconstruction, (*pointIter).costSmoothedPoint, (*pointIter).costLink);
			}
			fprintf_s(fp, "\t}\n}\n");			
		}
		
		fclose(fp);
	}
	catch (DWORD dwError)
	{
		printf("[ERROR](PrintTracks) cannot open file! error code %d\n", dwError);
		return;
	}
}

/************************************************************************
 Method Name: PrintHypotheses
 Description: 
	- print out information of tracks
 Input Arguments:
	- queueTracks: seletect tracks for print out
	- strFilePathAndName: output file path and name
	- bAppend: option for appending
 Return Values:
	- void
************************************************************************/
void CPSNWhere_Associator3D::PrintHypotheses(PSN_HypothesisSet &queueHypotheses, char *strFilePathAndName, unsigned int frameIdx)
{
	if (0 == queueHypotheses.size()) { return; }
	try
	{
		FILE *fp;
		std::string trackIDList;
		fopen_s(&fp, strFilePathAndName, "w");
		fprintf_s(fp, "frameIndex:%d\n", (int)frameIdx);
		fprintf_s(fp, "numHypotheses:%d\n", (int)queueHypotheses.size());
		int rank = 0;
		for (PSN_HypothesisSet::iterator hypothesisIter = queueHypotheses.begin();
			hypothesisIter != queueHypotheses.end();
			hypothesisIter++, rank++)
		{
			fprintf_s(fp, "{\n\trank:%d\n", rank);
			fprintf_s(fp, "\tselectedTracks:%d,", (int)(*hypothesisIter).selectedTracks.size());
			trackIDList = psn::MakeTrackIDList(&(*hypothesisIter).selectedTracks) + "\n";
			fprintf_s(fp, trackIDList.c_str());
			fprintf_s(fp, "\n\t{\n", (int)(*hypothesisIter).selectedTracks.size());
			for (int trackIdx = 0; trackIdx < (*hypothesisIter).selectedTracks.size(); trackIdx++)
			{
				Track3D *curTrack = (*hypothesisIter).selectedTracks[trackIdx];
				fprintf_s(fp, "\t\t{id:%d,Total:%e,Recon:%e,Link:%e,Enr:%e,Ex:%e,RGB:%e}\n",
					(int)curTrack->id, 
					curTrack->costTotal, 
					curTrack->costReconstruction, 
					curTrack->costLink,
					curTrack->costEnter,
					curTrack->costExit,
					curTrack->costRGB);
			}
			fprintf_s(fp, "\t}\n");

			fprintf_s(fp, "\trelatedTracks:%d,", (int)(*hypothesisIter).relatedTracks.size());
			trackIDList = psn::MakeTrackIDList(&(*hypothesisIter).relatedTracks) + "\n";
			fprintf_s(fp, trackIDList.c_str());

			fprintf_s(fp, "\tlogLikelihood:%e\n", (*hypothesisIter).logLikelihood);
			fprintf_s(fp, "\tprobability:%e\n", (*hypothesisIter).probability);
			fprintf_s(fp, "\tbValid:%d\n\t}\n", (int)(*hypothesisIter).bValid);
		}		
		fclose(fp);
	}
	catch (DWORD dwError)
	{
		printf("[ERROR](PrintHypotheses) cannot open file! error code %d\n", dwError);
		return;
	}
}

/************************************************************************
 Method Name: PrintCurrentTrackTrees
 Description: 
	- print out tree structure of current tracks
 Input Arguments:
	- strFilePathAndName: output file path and name
 Return Values:
	- void
************************************************************************/
void CPSNWhere_Associator3D::PrintCurrentTrackTrees(const char *strFilePath)
{
	try
	{
		FILE *fp;
		fopen_s(&fp, strFilePath, "w");
		int numTree = 0;

		// structure
		std::deque<stTrackInTreeInfo> queueNodes;
		for (std::list<TrackTree>::iterator treeIter = listTrackTree_.begin();
			treeIter != listTrackTree_.end();
			treeIter++)
		{
			if (0 == (*treeIter).tracks.size()) { continue;	}
			numTree++;

			Track3D* curTrack = (*treeIter).tracks.front();
			if (!curTrack->bValid) { continue; }
			stTrackInTreeInfo newInfo;
			newInfo.id = curTrack->id;
			newInfo.parentNode = 0;
			newInfo.timeGenerated = curTrack->timeGeneration;
			newInfo.GTP = (float)curTrack->GTProb;
			queueNodes.push_back(newInfo);
			TrackTree::MakeTreeNodesWithChildren(curTrack->childrenTrack, (int)queueNodes.size(), queueNodes);
		}

		// write trees
		fprintf(fp, "numTrees:%d\n", numTree);
		fprintf(fp, "nodeLength:%d,{", (int)queueNodes.size());
		for (std::deque<stTrackInTreeInfo>::iterator idxIter = queueNodes.begin();
			idxIter != queueNodes.end();
			idxIter++)
		{
			fprintf(fp, "(%d,%d,%d,%f)", (*idxIter).id, (*idxIter).parentNode, (*idxIter).timeGenerated, (*idxIter).GTP);
			if (idxIter < queueNodes.end() - 1) { fprintf(fp, ","); }
		}
		fprintf(fp, "}");
		
		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](PrintCurrentTrackTrees) cannot open file! error code %d\n", dwError);
		return;
	}
}


/************************************************************************
 Method Name: PrintResult
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::PrintResult(const char *strFilepath, std::deque<stTrack3DResult> *queueResults)
{
	FILE *fp;
	try
	{		
		fopen_s(&fp, strFilepath, "w");

		for(std::deque<stTrack3DResult>::iterator resultIter = queueResults->begin();
			resultIter != queueResults->end();
			resultIter++)
		{
			fprintf_s(fp, "{\n\tframeIndex:%d\n\tnumObjects:%d\n", (int)(*resultIter).frameIdx, (int)(*resultIter).object3DInfo.size());
			for(std::vector<stObject3DInfo>::iterator objectIter = (*resultIter).object3DInfo.begin();
				objectIter != (*resultIter).object3DInfo.end();
				objectIter++)
			{
				if (0 == (*objectIter).recentPoints.size()) { continue; }
				PSN_Point3D curPoint = (*objectIter).recentPoints.back();
				fprintf_s(fp, "\t{id:%d,position:(%f,%f,%f)}\n", (int)(*objectIter).id, (float)curPoint.x, (float)curPoint.y, (float)curPoint.z);
			}
			fprintf_s(fp, "}\n");
		}
		
		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](PrintResult) cannot open file! error code %d\n", dwError);
		return;
	}
}

///************************************************************************
// Method Name: SaveDefferedResult
// Description: 
//	- 
// Input Arguments:
//	- none
// Return Values:
//	- none
//************************************************************************/
//void CPSNWhere_Associator3D::SaveDefferedResult(unsigned int deferredLength)
//{
//	if(nCurrentFrameIdx_ < deferredLength)
//	{
//		return;
//	}
//
//	for(; nLastPrintedDeferredResultFrameIdx_ + deferredLength <= nCurrentFrameIdx_; nLastPrintedDeferredResultFrameIdx_++)
//	{
//		stTrack3DResult newResult;
//		newResult.frameIdx = nLastPrintedDeferredResultFrameIdx_;
//		newResult.processingTime = 0.0;
//		newResult.object3DInfo.clear();
//
//		for(std::deque<Track3D*>::iterator trackIter = queueTracksInBestSolution_.begin();
//			trackIter != queueTracksInBestSolution_.end();
//			trackIter++)
//		{
//			if((*trackIter)->timeEnd < nLastPrintedDeferredResultFrameIdx_ || (*trackIter)->timeStart > nLastPrintedDeferredResultFrameIdx_)
//			{
//				continue;
//			}
//
//			unsigned int curPosIdx = nLastPrintedDeferredResultFrameIdx_ - (*trackIter)->timeStart;
//
//			stObject3DInfo newObjectInfo;
//			newObjectInfo.id = (*trackIter)->tree->id;
//			newObjectInfo.recentPoints.push_back((*trackIter)->reconstructions[curPosIdx].point);
//
//			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
//			{
//				if(NULL == (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx))
//				{
//					newObjectInfo.bVisibleInViews[camIdx] = false;
//					newObjectInfo.rectInViews[camIdx] = PSN_Rect(0, 0, 0, 0);
//				}
//				else
//				{
//					newObjectInfo.bVisibleInViews[camIdx] = true;
//					unsigned int rectIdx = nLastPrintedDeferredResultFrameIdx_ - (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx)->timeStart;
//					newObjectInfo.rectInViews[camIdx] = (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx)->rects[rectIdx];					
//				}
//			}
//
//			newResult.object3DInfo.push_back(newObjectInfo);
//		}
//		queueDeferredTrackingResult_.push_back(newResult);
//	}
//}
//
//
///************************************************************************
// Method Name: SaveInstantResult
// Description: 
//	- 
// Input Arguments:
//	- none
// Return Values:
//	- none
//************************************************************************/
//void CPSNWhere_Associator3D::SaveInstantResult(void)
//{
//	stTrack3DResult newResult;
//	newResult.frameIdx = nCurrentFrameIdx_;
//	newResult.processingTime = fCurrentProcessingTime_;
//	newResult.object3DInfo.clear();
//
//	for(std::deque<Track3D*>::iterator trackIter = queueTracksInBestSolution_.begin();
//		trackIter != queueTracksInBestSolution_.end();
//		trackIter++)
//	{
//		if((*trackIter)->timeEnd < newResult.frameIdx || (*trackIter)->timeStart > newResult.frameIdx)
//		{
//			continue;
//		}
//
//		unsigned int curPosIdx = newResult.frameIdx - (*trackIter)->timeStart;
//
//		stObject3DInfo newObjectInfo;
//		newObjectInfo.id = (*trackIter)->tree->id;
//		newObjectInfo.recentPoints.push_back((*trackIter)->reconstructions[curPosIdx].point);
//
//		for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
//		{
//			if(NULL == (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx))
//			{
//				newObjectInfo.bVisibleInViews[camIdx] = false;
//				newObjectInfo.rectInViews[camIdx] = PSN_Rect(0, 0, 0, 0);
//			}
//			else
//			{
//				newObjectInfo.bVisibleInViews[camIdx] = true;
//				unsigned int rectIdx = newResult.frameIdx - (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx)->timeStart;
//				newObjectInfo.rectInViews[camIdx] = (*trackIter)->reconstructions[curPosIdx].tracklet2Ds.get(camIdx)->rects[rectIdx];					
//			}
//		}
//
//		newResult.object3DInfo.push_back(newObjectInfo);
//	}
//
//	queueTrackingResult_.push_back(newResult);
//}

/************************************************************************
 Method Name: SaveSnapshot
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Associator3D::SaveSnapshot(const char *strFilepath)
{
	FILE *fpTracklet2D, *fpTrack3D, *fpHypothesis, *fpResult, *fpInfo;
	char strFilename[128] = "";
	std::string trackIDList;
	try
	{	
		sprintf_s(strFilename, "%ssnapshot_3D_tracklet.txt", strFilepath);
		fopen_s(&fpTracklet2D, strFilename, "w");
		fprintf_s(fpTracklet2D, "numCamera:%d\n", NUM_CAM);
		fprintf_s(fpTracklet2D, "frameIndex:%d\n\n", (int)nCurrentFrameIdx_);
		
		//---------------------------------------------------------
		// 2D TRACKLET RELATED
		//---------------------------------------------------------
		fprintf_s(fpTracklet2D, "vecTracklet2DSet:\n{\n");
		for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			fprintf_s(fpTracklet2D, "\t{\n");
			fprintf_s(fpTracklet2D, "\t\ttracklets:%d,\n\t\t{\n", (int)vecTracklet2DSet_[camIdx].tracklets.size());
			for (std::list<stTracklet2D>::iterator trackletIter = vecTracklet2DSet_[camIdx].tracklets.begin();
				trackletIter != vecTracklet2DSet_[camIdx].tracklets.end();
				trackletIter++)
			{
				stTracklet2D *curTracklet = &(*trackletIter);
				fprintf_s(fpTracklet2D, "\t\t\t{\n");
				fprintf_s(fpTracklet2D, "\t\t\t\tid:%d\n", (int)curTracklet->id);
				fprintf_s(fpTracklet2D, "\t\t\t\tcamIdx:%d\n", (int)curTracklet->camIdx);
				fprintf_s(fpTracklet2D, "\t\t\t\tbActivated:%d\n", (int)curTracklet->bActivated);
				fprintf_s(fpTracklet2D, "\t\t\t\ttimeStart:%d\n", (int)curTracklet->timeStart);
				fprintf_s(fpTracklet2D, "\t\t\t\ttimeEnd:%d\n", (int)curTracklet->timeEnd);
				fprintf_s(fpTracklet2D, "\t\t\t\tduration:%d\n", (int)curTracklet->duration);

				// rects
				fprintf_s(fpTracklet2D, "\t\t\t\trects:%d,\n\t\t\t\t{\n", (int)curTracklet->rects.size());
				for (int rectIdx = 0; rectIdx < curTracklet->rects.size(); rectIdx++)
				{
					PSN_Rect curRect = curTracklet->rects[rectIdx];
					fprintf_s(fpTracklet2D, "\t\t\t\t\t(%f,%f,%f,%f)\n", (float)curRect.x, (float)curRect.y, (float)curRect.w, (float)curRect.h);
				}
				fprintf_s(fpTracklet2D, "\t\t\t\t}\n");

				// backprojectionLines
				fprintf_s(fpTracklet2D, "\t\t\t\tbackprojectionLines:%d,\n\t\t\t\t{\n", (int)curTracklet->backprojectionLines.size());
				for (int lineIdx = 0; lineIdx < curTracklet->backprojectionLines.size(); lineIdx++)
				{
					PSN_Point3D pt1 = curTracklet->backprojectionLines[lineIdx].first;
					PSN_Point3D pt2 = curTracklet->backprojectionLines[lineIdx].second;
					fprintf_s(fpTracklet2D, "\t\t\t\t\t(%f,%f,%f,%f,%f,%f)\n", (float)pt1.x, (float)pt1.y, (float)pt1.z, (float)pt2.x, (float)pt2.y, (float)pt2.z);
				}
				fprintf_s(fpTracklet2D, "\t\t\t\t}\n");

				// appearance
				fprintf_s(fpTracklet2D, "\t\t\t\tRGBFeatureHead:%d,(", curTracklet->RGBFeatureHead.rows);
				for (int idx = 0; idx < curTracklet->RGBFeatureHead.rows; idx++)
				{
					fprintf_s(fpTracklet2D, "%f", curTracklet->RGBFeatureHead.at<double>(idx, 0));
					if (idx < curTracklet->RGBFeatureHead.rows - 1) { fprintf_s(fpTracklet2D, ","); }
				}
				fprintf_s(fpTracklet2D, ")\n");
				fprintf_s(fpTracklet2D, "\t\t\t\tRGBFeatureTail:%d,(", curTracklet->RGBFeatureTail.rows);
				for (int idx = 0; idx < curTracklet->RGBFeatureTail.rows; idx++)
				{
					fprintf_s(fpTracklet2D, "%f", curTracklet->RGBFeatureTail.at<double>(idx, 0));
					if (idx < curTracklet->RGBFeatureTail.rows - 1) { fprintf_s(fpTracklet2D, ","); }
				}
				fprintf_s(fpTracklet2D, ")\n");

				// location in 3D
				fprintf_s(fpTracklet2D, "\t\t\t\tcurrentLocation3D:(%f,%f,%f)\n", curTracklet->currentLocation3D.x, curTracklet->currentLocation3D.y, curTracklet->currentLocation3D.z);

				// bAssociableNewMeasurement
				for (int compCamIdx = 0; compCamIdx < NUM_CAM; compCamIdx++)
				{
					fprintf_s(fpTracklet2D, "\t\t\t\tbAssociableNewMeasurement[%d]:%d,{", compCamIdx, (int)curTracklet->bAssociableNewMeasurement[compCamIdx].size());
					for (int flagIdx = 0; flagIdx < curTracklet->bAssociableNewMeasurement[compCamIdx].size(); flagIdx++)
					{
						fprintf_s(fpTracklet2D, "%d", (int)curTracklet->bAssociableNewMeasurement[compCamIdx][flagIdx]);
						if (flagIdx < curTracklet->bAssociableNewMeasurement[compCamIdx].size() - 1) { fprintf_s(fpTracklet2D, ","); }
					}
					fprintf_s(fpTracklet2D, "}\n");
				}
				fprintf_s(fpTracklet2D, "\t\t\t}\n");
			}

			fprintf_s(fpTracklet2D, "\t\tactiveTracklets:%d,{", (int)vecTracklet2DSet_[camIdx].activeTracklets.size());
			for (std::deque<stTracklet2D*>::iterator trackletIter = vecTracklet2DSet_[camIdx].activeTracklets.begin();
				trackletIter != vecTracklet2DSet_[camIdx].activeTracklets.end();
				trackletIter++)
			{
				fprintf_s(fpTracklet2D, "%d", (int)(*trackletIter)->id);
				if (trackletIter != vecTracklet2DSet_[camIdx].activeTracklets.end() - 1) { fprintf_s(fpTracklet2D, ","); }
			}
			fprintf_s(fpTracklet2D, "}\n\t\tnewMeasurements:%d,{", (int)vecTracklet2DSet_[camIdx].newMeasurements.size());
			for (std::deque<stTracklet2D*>::iterator trackletIter = vecTracklet2DSet_[camIdx].newMeasurements.begin();
				trackletIter != vecTracklet2DSet_[camIdx].newMeasurements.end();
				trackletIter++)
			{
				fprintf_s(fpTracklet2D, "%d", (int)(*trackletIter)->id);
				if (trackletIter != vecTracklet2DSet_[camIdx].newMeasurements.end() - 1) { fprintf_s(fpTracklet2D, ","); }
			}
			fprintf_s(fpTracklet2D, "}\n\t}\n");
		}
		fprintf_s(fpTracklet2D, "}\n");
		fprintf_s(fpTracklet2D, "nNumTotalActive2DTracklet:%d\n\n", (int)nNumTotalActive2DTracklet_);
		fclose(fpTracklet2D);

		//---------------------------------------------------------
		// 3D TRACK RELATED
		//---------------------------------------------------------
		sprintf_s(strFilename, "%ssnapshot_3D_track.txt", strFilepath);
		fopen_s(&fpTrack3D, strFilename, "w");
		fprintf_s(fpTrack3D, "numCamera:%d\n", NUM_CAM);
		fprintf_s(fpTrack3D, "frameIndex:%d\n\n", (int)nCurrentFrameIdx_);
		fprintf_s(fpTrack3D, "bReceiveNewMeasurement:%d\n", (int)bReceiveNewMeasurement_);
		fprintf_s(fpTrack3D, "bInitiationPenaltyFree:%d\n", (int)bInitiationPenaltyFree_);
		fprintf_s(fpTrack3D, "nNewTrackID:%d\n", (int)nNewTrackID_);
		fprintf_s(fpTrack3D, "nNewTreeID:%d\n", (int)nNewTreeID_);

		// track
		fprintf_s(fpTrack3D, "listTrack3D:%d,\n{\n", listTrack3D_.size());
		for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
			trackIter != listTrack3D_.end();
			trackIter++)
		{
			Track3D *curTrack = &(*trackIter);			
			fprintf_s(fpTrack3D, "\t{\n\t\tid:%d\n", (int)curTrack->id);

			// curTracklet2Ds
			fprintf_s(fpTrack3D, "\t\tcurTracklet2Ds:{");
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				if (NULL == curTrack->curTracklet2Ds.get(camIdx)) 
				{ 
					fprintf_s(fpTrack3D, "-1"); 
				}
				else 
				{ 
					fprintf_s(fpTrack3D, "%d", curTrack->curTracklet2Ds.get(camIdx)->id); 
				}

				if (camIdx < NUM_CAM - 1) 
				{ 
					fprintf_s(fpTrack3D, ","); 
				}
			}
			fprintf_s(fpTrack3D, "}\n");

			// tracklet2DIDs
			fprintf_s(fpTrack3D, "\t\ttrackleIDs:\n\t\t{\n");
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				if(0 == curTrack->tracklet2DIDs[camIdx].size())
				{
					fprintf_s(fpTrack3D, "\t\t\tnumTracklet:0,{}\n");
					continue;
				}
				fprintf_s(fpTrack3D, "\t\t\tnumTracklet:%d,{", (int)curTrack->tracklet2DIDs[camIdx].size());
				for(unsigned int trackletIDIdx = 0; trackletIDIdx < curTrack->tracklet2DIDs[camIdx].size(); trackletIDIdx++)
				{
					fprintf_s(fpTrack3D, "%d", (int)curTrack->tracklet2DIDs[camIdx][trackletIDIdx]);
					if (curTrack->tracklet2DIDs[camIdx].size() - 1 != trackletIDIdx) { fprintf_s(fpTrack3D, ","); }
					else { fprintf_s(fpTrack3D, "}\n");	}
				}
			}
			fprintf_s(fpTrack3D, "\t\t}\n");

			// tracklet last locations
			fprintf_s(fpTrack3D, "\t\ttrackletLastLocation3D:{");
			for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				fprintf_s(fpTrack3D, "(%f,%f,%f)", 
					curTrack->trackletLastLocation3D[camIdx].x,
					curTrack->trackletLastLocation3D[camIdx].y,
					curTrack->trackletLastLocation3D[camIdx].z);
			}
			fprintf_s(fpTrack3D, "}\n");

			fprintf_s(fpTrack3D, "\t\tbActive:%d\n", (int)curTrack->bActive);
			fprintf_s(fpTrack3D, "\t\tbValid:%d\n", (int)curTrack->bValid);
			fprintf_s(fpTrack3D, "\t\ttreeID:%d\n", (int)curTrack->tree->id);
			if (NULL == curTrack->parentTrack) { fprintf_s(fpTrack3D, "\t\tparentTrackID:-1\n"); }
			else { fprintf_s(fpTrack3D, "\t\tparentTrackID:%d\n", (int)curTrack->parentTrack->id); }

			// children tracks
			fprintf_s(fpTrack3D, "\t\tchildrenTrack:%d,", (int)curTrack->childrenTrack.size());
			trackIDList = psn::MakeTrackIDList(&curTrack->childrenTrack) + "\n";
			fprintf_s(fpTrack3D, trackIDList.c_str());

			// temporal information
			fprintf_s(fpTrack3D, "\t\ttimeStart:%d\n", (int)curTrack->timeStart);
			fprintf_s(fpTrack3D, "\t\ttimeEnd:%d\n", (int)curTrack->timeEnd);
			fprintf_s(fpTrack3D, "\t\ttimeGeneration:%d\n", (int)curTrack->timeGeneration);
			fprintf_s(fpTrack3D, "\t\tduration:%d\n", (int)curTrack->duration);

			// reconstrcution related
			fprintf(fpTrack3D, "\t\treconstructions:%d,\n\t\t{\n", (int)curTrack->reconstructions.size());
			for(std::deque<stReconstruction>::iterator pointIter = curTrack->reconstructions.begin();
				pointIter != curTrack->reconstructions.end();
				pointIter++)
			{
				fprintf_s(fpTrack3D, "\t\t\t%d,point:(%f,%f,%f),smoothedPoint:(%f,%f,%f),velocity:(%f,%f,%f),maxError:%e,costReconstruction:%e,costSmoothedPoint:%e,costLink:%e,", (int)(*pointIter).bIsMeasurement, 
					(*pointIter).point.x, (*pointIter).point.y, (*pointIter).point.z,
					(*pointIter).smoothedPoint.x, (*pointIter).smoothedPoint.y, (*pointIter).smoothedPoint.z,
					(*pointIter).velocity.x, (*pointIter).velocity.y, (*pointIter).velocity.z,
					(*pointIter).maxError, (*pointIter).costReconstruction, (*pointIter).costSmoothedPoint, (*pointIter).costLink);
				fprintf_s(fpTrack3D, "tracklet2Ds:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					if (NULL == (*pointIter).tracklet2Ds.get(camIdx)) { fprintf_s(fpTrack3D, "-1"); }
					else { fprintf_s(fpTrack3D, "%d", (*pointIter).tracklet2Ds.get(camIdx)->id); }
					if (camIdx < NUM_CAM - 1) { fprintf_s(fpTrack3D, ","); }
				}
				fprintf_s(fpTrack3D, "},");
				fprintf_s(fpTrack3D, "rawPoints:%d,{", (int)(*pointIter).rawPoints.size());
				for (int pointIdx = 0; pointIdx < (*pointIter).rawPoints.size(); pointIdx++)
				{
					fprintf_s(fpTrack3D, "(%f,%f,%f)", (*pointIter).rawPoints[pointIdx].x, (*pointIter).rawPoints[pointIdx].y, (*pointIter).rawPoints[pointIdx].z);
					if (pointIdx < (*pointIter).rawPoints.size() - 1) { fprintf_s(fpTrack3D, ","); }
				}
				fprintf_s(fpTrack3D, "}\n");
			}
			fprintf_s(fpTrack3D, "\t\t}\n");

			// point smoother
			fprintf_s(fpTrack3D, "\t\tsmoother:{span:%d,degree:%d}\n", SGS_DEFAULT_SPAN, SGS_DEFAULT_DEGREE);

			// cost
			fprintf_s(fpTrack3D, "\t\tcostTotal:%e\n", curTrack->costTotal);
			fprintf_s(fpTrack3D, "\t\tcostReconstruction:%e\n", curTrack->costReconstruction);
			fprintf_s(fpTrack3D, "\t\tcostLink:%e\n", curTrack->costLink);
			fprintf_s(fpTrack3D, "\t\tcostEnter:%e\n", curTrack->costEnter);
			fprintf_s(fpTrack3D, "\t\tcostExit:%e\n", curTrack->costExit);
			fprintf_s(fpTrack3D, "\t\tcostRGB:%e\n", curTrack->costRGB);

			// loglikelihood
			fprintf_s(fpTrack3D, "\t\tloglikelihood:%e\n", curTrack->loglikelihood);

			// GTP
			fprintf_s(fpTrack3D, "\t\tGTProb:%e\n", curTrack->GTProb);
			fprintf_s(fpTrack3D, "\t\tBranchGTProb:%e\n", curTrack->BranchGTProb);
			fprintf_s(fpTrack3D, "\t\tbWasBestSolution:%d\n", (int)curTrack->bWasBestSolution);
			fprintf_s(fpTrack3D, "\t\tbCurrentBestSolution:%d\n", (int)curTrack->bCurrentBestSolution);

			// HO-MHT
			fprintf_s(fpTrack3D, "\t\tbNewTrack:%d\n", (int)curTrack->bNewTrack);

			// appearance
			fprintf_s(fpTrack3D, "\t\tlastRGBFeature:\n\t\t{\n");
			for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				fprintf_s(fpTrack3D, "\t\t\tlastRGBFeature[%d]:%d,(", camIdx, curTrack->lastRGBFeature[camIdx].rows);
				for (int idx = 0; idx < curTrack->lastRGBFeature[camIdx].rows; idx++)
				{
					fprintf_s(fpTrack3D, "%f", curTrack->lastRGBFeature[camIdx].at<double>(idx, 0));
					if (idx < curTrack->lastRGBFeature[camIdx].rows - 1) { fprintf_s(fpTrack3D, ","); }
				}
				fprintf_s(fpTrack3D, ")\n");
			}
			fprintf_s(fpTrack3D, "\t\t}\n");
			fprintf_s(fpTrack3D, "\t\ttimeTrackletEnded:(");
			for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				fprintf_s(fpTrack3D, "%d", (int)curTrack->timeTrackletEnded[camIdx]);
				if (camIdx < NUM_CAM - 1) { fprintf_s(fpTrack3D, ","); }				
			}
			fprintf_s(fpTrack3D, ")\n\t}\n");
		}
		fprintf_s(fpTrack3D, "}\n");
		
		// track tree
		fprintf_s(fpTrack3D, "listTrackTree:%d,\n{\n", listTrackTree_.size());
		for (std::list<TrackTree>::iterator treeIter = listTrackTree_.begin();
			treeIter != listTrackTree_.end();
			treeIter++)
		{
			TrackTree *curTree = &(*treeIter);
			fprintf_s(fpTrack3D, "\t{\n\t\tid:%d\n", (int)curTree->id);
			fprintf_s(fpTrack3D, "\t\ttimeGeneration:%d\n", (int)curTree->timeGeneration);
			fprintf_s(fpTrack3D, "\t\tbValid:%d\n", (int)curTree->bValid);
			fprintf_s(fpTrack3D, "\t\ttracks:%d,", (int)curTree->tracks.size());
			trackIDList = psn::MakeTrackIDList(&curTree->tracks) + "\n";
			fprintf_s(fpTrack3D, trackIDList.c_str());
			//fprintf_s(fpTrack3D, "\t\tnumMeasurements:%d\n", (int)curTree->numMeasurements);
			//// tracklet2Ds
			//fprintf_s(fpTrack3D, "\t\ttrackle2Ds:\n\t\t{\n");
			//for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			//{
			//	if (0 == curTree->tracklet2Ds[camIdx].size())
			//	{
			//		fprintf_s(fpTrack3D, "\t\t\ttrackletInfo:0,{}\n");
			//		continue;
			//	}
			//	fprintf_s(fpTrack3D, "\t\t\ttrackletInfo:%d,\n\t\t\t{", (int)curTree->tracklet2Ds[camIdx].size());
			//	for (int trackletIDIdx = 0; trackletIDIdx < curTree->tracklet2Ds[camIdx].size(); trackletIDIdx++)
			//	{
			//		fprintf_s(fpTrack3D, "\n\t\t\t\tid:%d,relatedTracks:%d,{", (int)curTree->tracklet2Ds[camIdx][trackletIDIdx].tracklet2D->id, (int)curTree->tracklet2Ds[camIdx][trackletIDIdx].queueRelatedTracks.size());
			//		for (int trackIdx = 0; trackIdx < curTree->tracklet2Ds[camIdx][trackletIDIdx].queueRelatedTracks.size(); trackIdx++)
			//		{
			//			fprintf_s(fpTrack3D, "%d", curTree->tracklet2Ds[camIdx][trackletIDIdx].queueRelatedTracks[trackIdx]->id);
			//			if(curTree->tracklet2Ds[camIdx][trackletIDIdx].queueRelatedTracks.size() - 1 >  trackIdx)
			//			{
			//				fprintf_s(fpTrack3D, ",");
			//			}					
			//		}
			//		fprintf_s(fpTrack3D, "}\n");
			//	}
			//	fprintf_s(fpTrack3D, "\t\t\t}\n");
			//}
			//fprintf_s(fpTrack3D, "\t\t}\n\t}\n");
			fprintf_s(fpTrack3D, "\t}\n");
		}
		fprintf_s(fpTrack3D, "}\n");

		// new seed tracks
		fprintf_s(fpTrack3D, "queueNewSeedTracks:%d,", (int)queueNewSeedTracks_.size());
		trackIDList = psn::MakeTrackIDList(&queueNewSeedTracks_) + "\n";
		fprintf_s(fpTrack3D, trackIDList.c_str());

		// active tracks
		fprintf_s(fpTrack3D, "queueActiveTrack:%d,", (int)queueActiveTrack_.size());
		trackIDList = psn::MakeTrackIDList(&queueActiveTrack_) + "\n";
		fprintf_s(fpTrack3D, trackIDList.c_str());

		// puased tracks
		fprintf_s(fpTrack3D, "queuePausedTrack:%d,", (int)queuePausedTrack_.size());
		trackIDList = psn::MakeTrackIDList(&queuePausedTrack_) + "\n";
		fprintf_s(fpTrack3D, trackIDList.c_str());

		// tracks in window
		fprintf_s(fpTrack3D, "queueTracksInWindow:%d,", (int)queueTracksInWindow_.size());
		trackIDList = psn::MakeTrackIDList(&queueTracksInWindow_) + "\n";
		fprintf_s(fpTrack3D, trackIDList.c_str());

		// tracks in the best solution
		fprintf_s(fpTrack3D, "queueTracksInBestSolution:%d,", (int)queueTracksInBestSolution_.size());
		trackIDList = psn::MakeTrackIDList(&queueTracksInBestSolution_) + "\n";
		fprintf_s(fpTrack3D, trackIDList.c_str());

		// active trees
		fprintf_s(fpTrack3D, "queuePtActiveTrees:%d,{", (int)queuePtActiveTrees_.size());
		for (int treeIdx = 0; treeIdx < queuePtActiveTrees_.size(); treeIdx++)
		{
			fprintf_s(fpTrack3D, "%d", queuePtActiveTrees_[treeIdx]->id);
			if (treeIdx < queuePtActiveTrees_.size() - 1) { fprintf_s(fpTrack3D, ","); }
		}
		fprintf_s(fpTrack3D, "}\n");

		// unconfirmed trees
		fprintf_s(fpTrack3D, "queuePtUnconfirmedTrees:%d,{", (int)queuePtUnconfirmedTrees_.size());
		for (int treeIdx = 0; treeIdx < queuePtUnconfirmedTrees_.size(); treeIdx++)
		{
			fprintf_s(fpTrack3D, "%d", queuePtUnconfirmedTrees_[treeIdx]->id);
			if (treeIdx < queuePtUnconfirmedTrees_.size() - 1) { fprintf_s(fpTrack3D, ","); }
		}
		fprintf_s(fpTrack3D, "}\n");
		fclose(fpTrack3D);

		//---------------------------------------------------------
		// RESULTS
		//---------------------------------------------------------
		sprintf_s(strFilename, "%ssnapshot_3D_result.txt", strFilepath);
		fopen_s(&fpResult, strFilename, "w");
		fprintf_s(fpResult, "numCamera:%d\n", NUM_CAM);
		fprintf_s(fpResult, "frameIndex:%d\n\n", (int)nCurrentFrameIdx_);

		// instance result
		fprintf_s(fpResult, "queueTrackingResult:%d,\n{\n", (int)queueTrackingResult_.size());
		for (int resultIdx = 0; resultIdx < queueTrackingResult_.size(); resultIdx++)
		{
			stTrack3DResult *curResult = &queueTrackingResult_[resultIdx];
			fprintf_s(fpResult, "\t{\n\t\tframeIdx:%d\n", curResult->frameIdx);
			fprintf_s(fpResult, "\t\tprocessingTime:%f\n", curResult->processingTime);			

			// object info
			fprintf_s(fpResult, "\t\tobjectInfo:%d,\n\t\t{\n", (int)curResult->object3DInfo.size());
			for (int objIdx = 0; objIdx < curResult->object3DInfo.size(); objIdx++)
			{
				stObject3DInfo *curObject = &curResult->object3DInfo[objIdx];
				fprintf_s(fpResult, "\t\t\t{\n\t\t\t\tid:%d\n", curObject->id);

				// points
				fprintf_s(fpResult, "\t\t\t\trecentPoints:%d,{", (int)curObject->recentPoints.size());
				for (int pointIdx = 0; pointIdx < curObject->recentPoints.size(); pointIdx++)
				{
					PSN_Point3D curPoint = curObject->recentPoints[pointIdx];
					fprintf_s(fpResult, "(%f,%f,%f)", (float)curPoint.x, (float)curPoint.y, (float)curPoint.z);
					if (pointIdx < curObject->recentPoints.size() - 1) { fprintf_s(fpResult, ","); }
				}
				fprintf_s(fpResult, "}\n");

				// 2D points
				fprintf_s(fpResult, "\t\t\t\trecentPoint2Ds:\n\t\t\t\t{\n");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{					
					fprintf_s(fpResult, "\t\t\t\t\tcam%d:%d,{", camIdx, (int)curObject->recentPoint2Ds[camIdx].size());
					for (int pointIdx = 0; pointIdx < curObject->recentPoint2Ds[camIdx].size(); pointIdx++)
					{
						PSN_Point2D curPoint = curObject->recentPoint2Ds[camIdx][pointIdx];
						fprintf_s(fpResult, "(%f,%f)", (float)curPoint.x, (float)curPoint.y);
						if (pointIdx < curObject->recentPoint2Ds[camIdx].size() - 1) { fprintf_s(fpResult, ","); }
					}
					fprintf_s(fpResult, "}\n");
				}
				fprintf_s(fpResult, "\t\t\t\t}\n");

				// 3D box points in each view
				fprintf_s(fpResult, "\t\t\t\tpoint3DBox:\n\t\t\t\t{\n");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{					
					fprintf_s(fpResult, "\t\t\t\t\tcam%d:%d,{", camIdx, (int)curObject->point3DBox[camIdx].size());
					for (int pointIdx = 0; pointIdx < curObject->point3DBox[camIdx].size(); pointIdx++)
					{
						PSN_Point2D curPoint = curObject->point3DBox[camIdx][pointIdx];
						fprintf_s(fpResult, "(%f,%f)", (float)curPoint.x, (float)curPoint.y);
						if (pointIdx < curObject->recentPoint2Ds[camIdx].size() - 1) { fprintf_s(fpResult, ","); }
					}
					fprintf_s(fpResult, "}\n");
				}
				fprintf_s(fpResult, "\t\t\t\t}\n");

				// rects
				fprintf_s(fpResult, "\t\t\t\trectInViews:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					PSN_Rect curRect = curObject->rectInViews[camIdx];
					fprintf_s(fpResult, "(%f,%f,%f,%f)", (float)curRect.x, (float)curRect.y, (float)curRect.w, (float)curRect.h);
					if (camIdx < NUM_CAM - 1) { fprintf_s(fpResult, ","); }
				}
				fprintf_s(fpResult, "}\n");

				// visibility
				fprintf_s(fpResult, "\t\t\t\tbVisibleInViews:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					fprintf_s(fpResult, "%d", (int)curObject->bVisibleInViews[camIdx]);
					if (camIdx < NUM_CAM - 1) { fprintf_s(fpResult, ","); }
				}
				fprintf_s(fpResult, "}\n\t\t\t}\n");
			}
			fprintf_s(fpResult, "\t\t}\n\t}\n");
		}
		fprintf_s(fpResult, "}\n");

		// deferred result
		fprintf_s(fpResult, "queueDeferredTrackingResult:%d,\n{\n", (int)queueDeferredTrackingResult_.size());
		for (int resultIdx = 0; resultIdx < queueDeferredTrackingResult_.size(); resultIdx++)
		{
			stTrack3DResult *curResult = &queueDeferredTrackingResult_[resultIdx];
			fprintf_s(fpResult, "\t{\n\t\tframeIdx:%d\n", curResult->frameIdx);
			fprintf_s(fpResult, "\t\tprocessingTime:%f\n", curResult->processingTime);			

			// object info
			fprintf_s(fpResult, "\t\tobjectInfo:%d,\n\t\t{\n", (int)curResult->object3DInfo.size());
			for (int objIdx = 0; objIdx < curResult->object3DInfo.size(); objIdx++)
			{
				stObject3DInfo *curObject = &curResult->object3DInfo[objIdx];
				fprintf_s(fpResult, "\t\t\t{\n\t\t\t\tid:%d\n", curObject->id);

				// points
				fprintf_s(fpResult, "\t\t\t\trecentPoints:%d,{", (int)curObject->recentPoints.size());
				for (int pointIdx = 0; pointIdx < curObject->recentPoints.size(); pointIdx++)
				{
					PSN_Point3D curPoint = curObject->recentPoints[pointIdx];
					fprintf_s(fpResult, "(%f,%f,%f)", (float)curPoint.x, (float)curPoint.y, (float)curPoint.z);
					if (pointIdx < curObject->recentPoints.size() - 1) { fprintf_s(fpResult, ","); }
				}
				fprintf_s(fpResult, "}\n");

				// 2D points
				fprintf_s(fpResult, "\t\t\t\trecentPoint2Ds:\n\t\t\t\t{\n");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{					
					fprintf_s(fpResult, "\t\t\t\t\tcam%d:%d,{", camIdx, (int)curObject->recentPoint2Ds[camIdx].size());
					for (int pointIdx = 0; pointIdx < curObject->recentPoint2Ds[camIdx].size(); pointIdx++)
					{
						PSN_Point2D curPoint = curObject->recentPoint2Ds[camIdx][pointIdx];
						fprintf_s(fpResult, "(%f,%f)", (float)curPoint.x, (float)curPoint.y);
						if (pointIdx < curObject->recentPoint2Ds[camIdx].size() - 1) { fprintf_s(fpResult, ","); }
					}
					fprintf_s(fpResult, "}\n");
				}
				fprintf_s(fpResult, "\t\t\t\t}\n");

				// 3D box points in each view
				fprintf_s(fpResult, "\t\t\t\tpoint3DBox:\n\t\t\t\t{\n");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{					
					fprintf_s(fpResult, "\t\t\t\t\tcam%d:%d,{", camIdx, (int)curObject->point3DBox[camIdx].size());
					for (int pointIdx = 0; pointIdx < curObject->point3DBox[camIdx].size(); pointIdx++)
					{
						PSN_Point2D curPoint = curObject->point3DBox[camIdx][pointIdx];
						fprintf_s(fpResult, "(%f,%f)", (float)curPoint.x, (float)curPoint.y);
						if (pointIdx < curObject->recentPoint2Ds[camIdx].size() - 1) { fprintf_s(fpResult, ","); }
					}
					fprintf_s(fpResult, "}\n");
				}
				fprintf_s(fpResult, "\t\t\t\t}\n");

				// rects
				fprintf_s(fpResult, "\t\t\t\trectInViews:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					PSN_Rect curRect = curObject->rectInViews[camIdx];
					fprintf_s(fpResult, "(%f,%f,%f,%f)", (float)curRect.x, (float)curRect.y, (float)curRect.w, (float)curRect.h);
					if (camIdx < NUM_CAM - 1) { fprintf_s(fpResult, ","); }
				}
				fprintf_s(fpResult, "}\n");

				// visibility
				fprintf_s(fpResult, "\t\t\t\tbVisibleInViews:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					fprintf_s(fpResult, "%d", (int)curObject->bVisibleInViews[camIdx]);
					if (camIdx < NUM_CAM - 1) { fprintf_s(fpResult, ","); }
				}
				fprintf_s(fpResult, "}\n\t\t\t}\n");
			}
			fprintf_s(fpResult, "\t\t}\n\t}\n");
		}
		fprintf_s(fpResult, "}\n");
		fclose(fpResult);

		//---------------------------------------------------------
		// HYPOTHESES
		//---------------------------------------------------------
		sprintf_s(strFilename, "%ssnapshot_3D_hypotheses.txt", strFilepath);
		fopen_s(&fpHypothesis, strFilename, "w");
		fprintf_s(fpHypothesis, "numCamera:%d\n", NUM_CAM);
		fprintf_s(fpHypothesis, "frameIndex:%d\n\n", (int)nCurrentFrameIdx_);

		// queuePrevGlobalHypotheses
		fprintf_s(fpHypothesis, "queuePrevGlobalHypotheses:%d,\n{\n", (int)queuePrevGlobalHypotheses_.size());
		for (PSN_HypothesisSet::iterator hypothesisIter = queuePrevGlobalHypotheses_.begin();
			hypothesisIter != queuePrevGlobalHypotheses_.end();
			hypothesisIter++)
		{
			fprintf_s(fpHypothesis, "\t{\n\t\tselectedTracks:%d,", (int)(*hypothesisIter).selectedTracks.size());
			trackIDList = psn::MakeTrackIDList(&(*hypothesisIter).selectedTracks) + "\n";
			fprintf_s(fpHypothesis, trackIDList.c_str());

			fprintf_s(fpHypothesis, "\t\trelatedTracks:%d,", (int)(*hypothesisIter).relatedTracks.size());
			trackIDList = psn::MakeTrackIDList(&(*hypothesisIter).relatedTracks) + "\n";
			fprintf_s(fpHypothesis, trackIDList.c_str());

			fprintf_s(fpHypothesis, "\t\tlogLikelihood:%e\n", (*hypothesisIter).logLikelihood);
			fprintf_s(fpHypothesis, "\t\tprobability:%e\n", (*hypothesisIter).probability);
			fprintf_s(fpHypothesis, "\t\tbValid:%d\n\t}\n", (int)(*hypothesisIter).bValid);
		}
		fprintf_s(fpHypothesis, "}\n");

		// queueCurrGlobalHypotheses
		fprintf_s(fpHypothesis, "queueCurrGlobalHypotheses:%d,\n{\n", (int)queueCurrGlobalHypotheses_.size());
		for (PSN_HypothesisSet::iterator hypothesisIter = queueCurrGlobalHypotheses_.begin();
			hypothesisIter != queueCurrGlobalHypotheses_.end();
			hypothesisIter++)
		{
			fprintf_s(fpHypothesis, "\t{\n\t\tselectedTracks:%d,", (int)(*hypothesisIter).selectedTracks.size());
			trackIDList = psn::MakeTrackIDList(&(*hypothesisIter).selectedTracks) + "\n";
			fprintf_s(fpHypothesis, trackIDList.c_str());

			fprintf_s(fpHypothesis, "\t\trelatedTracks:%d,", (int)(*hypothesisIter).relatedTracks.size());
			trackIDList = psn::MakeTrackIDList(&(*hypothesisIter).relatedTracks) + "\n";
			fprintf_s(fpHypothesis, trackIDList.c_str());

			fprintf_s(fpHypothesis, "\t\tlogLikelihood:%e\n", (*hypothesisIter).logLikelihood);
			fprintf_s(fpHypothesis, "\t\tprobability:%e\n", (*hypothesisIter).probability);
			fprintf_s(fpHypothesis, "\t\tbValid:%d\n\t}\n", (int)(*hypothesisIter).bValid);
		}
		fprintf_s(fpHypothesis, "}\n");
		fclose(fpHypothesis);

		//---------------------------------------------------------
		// VISUALIZATION
		//---------------------------------------------------------
		sprintf_s(strFilename, "%ssnapshot_3D_info.txt", strFilepath);
		fopen_s(&fpInfo, strFilename, "w");
		fprintf_s(fpInfo, "numCamera:%d\n", NUM_CAM);
		fprintf_s(fpInfo, "frameIndex:%d\n\n", (int)nCurrentFrameIdx_);
		fprintf_s(fpInfo, "nNewVisualizationID:%d\n", (int)nNewVisualizationID_);
		fprintf_s(fpInfo, "queuePairTreeIDToVisualizationID:%d,{", (int)queuePairTreeIDToVisualizationID_.size());
		for (int pairIdx = 0; pairIdx < queuePairTreeIDToVisualizationID_.size(); pairIdx++)
		{
			fprintf_s(fpInfo, "(%d,%d)", queuePairTreeIDToVisualizationID_[pairIdx].first, queuePairTreeIDToVisualizationID_[pairIdx].second);
		}
		fprintf_s(fpInfo, "}\n");

		fprintf_s(fpInfo, "()()\n");
		fprintf_s(fpInfo, "('')\n");

		fclose(fpInfo);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](SaveSnapShot) cannot open file! error code %d\n", dwError);
		return;
	}
}

/************************************************************************
 Method Name: LoadSnapshot
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- 
************************************************************************/
bool CPSNWhere_Associator3D::LoadSnapshot(const char *strFilepath)
{
#ifdef PSN_DEBUG_MODE_
	printf(">> Reading snapshot......\n");
#endif
	FILE *fpTracklet, *fpTrack, *fpHypothesis, *fpResult, *fpInfo;
	char strFilename[128] = "";
	int readingInt = 0;
	float readingFloat = 0.0;

	std::deque<std::pair<Track3D*, unsigned int>> treeIDPair;
	std::deque<std::pair<Track3D*, std::deque<unsigned int>>> childrenTrackIDPair;

	try
	{	
		// file open
		sprintf_s(strFilename, "%ssnapshot_3D_tracklet.txt", strFilepath);		
		fopen_s(&fpTracklet, strFilename, "r"); 
		if (NULL == fpTracklet) { return false; }
		fscanf_s(fpTracklet, "numCamera:%d\n", &readingInt);
		assert(NUM_CAM == readingInt);
		fscanf_s(fpTracklet, "frameIndex:%d\n\n", &readingInt);
		nCurrentFrameIdx_ = (unsigned int)readingInt;
	
		//---------------------------------------------------------
		// 2D TRACKLET RELATED
		//---------------------------------------------------------
		fscanf_s(fpTracklet, "vecTracklet2DSet:\n{\n");
		for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
#ifdef PSN_DEBUG_MODE_
			printf(">> Reading 2D tracklet set at cam %d: %3.1f%%", camIdx, 0);
#endif
			int numTrackletSet = 0;
			fscanf_s(fpTracklet, "\t{\n\t\ttracklets:%d,\n\t\t{\n", &numTrackletSet);
			vecTracklet2DSet_[camIdx].tracklets.clear();
			for (int trackletIdx = 0; trackletIdx < numTrackletSet; trackletIdx++)
			{
				stTracklet2D newTracklet;

				fscanf_s(fpTracklet, "\t\t\t{\n");
				fscanf_s(fpTracklet, "\t\t\t\tid:%d\n", &readingInt);
				newTracklet.id = (unsigned int)readingInt;
				fscanf_s(fpTracklet, "\t\t\t\tcamIdx:%d\n", &readingInt);
				newTracklet.camIdx = (unsigned int)readingInt;
				fscanf_s(fpTracklet, "\t\t\t\tbActivated:%d\n", &readingInt);
				newTracklet.bActivated = 0 < readingInt ? true : false;
				fscanf_s(fpTracklet, "\t\t\t\ttimeStart:%d\n", &readingInt);
				newTracklet.timeStart = (unsigned int)readingInt;
				fscanf_s(fpTracklet, "\t\t\t\ttimeEnd:%d\n", &readingInt);
				newTracklet.timeEnd = (unsigned int)readingInt;
				fscanf_s(fpTracklet, "\t\t\t\tduration:%d\n", &readingInt);
				newTracklet.duration = (unsigned int)readingInt;

				// rects
				int numRect = 0;
				fscanf_s(fpTracklet, "\t\t\t\trects:%d,\n\t\t\t\t{\n", &numRect);
				for (int rectIdx = 0; rectIdx < numRect; rectIdx++)
				{
					float x, y, w, h;
					fscanf_s(fpTracklet, "\t\t\t\t\t(%f,%f,%f,%f)\n", &x, &y, &w, &h);
					newTracklet.rects.push_back(PSN_Rect((double)x, (double)y, (double)w, (double)h));
				}
				fscanf_s(fpTracklet, "\t\t\t\t}\n");

				// backprojectionLines
				int numLine = 0;
				fscanf_s(fpTracklet, "\t\t\t\tbackprojectionLines:%d,\n\t\t\t\t{\n", &numLine);
				for (int lineIdx = 0; lineIdx < numLine; lineIdx++)
				{
					float x1, y1, z1, x2, y2, z2;					
					fscanf_s(fpTracklet, "\t\t\t\t\t(%f,%f,%f,%f,%f,%f)\n", &x1, &y1, &z1, &x2, &y2, &z2);
					newTracklet.backprojectionLines.push_back(std::make_pair(PSN_Point3D((double)x1, (double)y1, (double)z1), PSN_Point3D((double)x2, (double)y2, (double)z2)));
				}
				fscanf_s(fpTracklet, "\t\t\t\t}\n");

				// appearance
				fscanf_s(fpTracklet, "\t\t\t\tRGBFeatureHead:%d,(", &readingInt);
				newTracklet.RGBFeatureHead = cv::Mat::zeros(readingInt, 1, CV_64FC1);
				for (int idx = 0; idx < readingInt; idx++)
				{
					fscanf_s(fpTracklet, "%f", &readingFloat);
					newTracklet.RGBFeatureHead.at<double>(idx, 0) = (double)readingFloat;
					if (idx < readingInt - 1) { fscanf_s(fpTracklet, ","); }
				}
				fscanf_s(fpTracklet, ")\n");
				fscanf_s(fpTracklet, "\t\t\t\tRGBFeatureTail:%d,(", &readingInt);
				newTracklet.RGBFeatureTail = cv::Mat::zeros(readingInt, 1, CV_64FC1);
				for (int idx = 0; idx < readingInt; idx++)
				{
					fscanf_s(fpTracklet, "%f", &readingFloat);
					newTracklet.RGBFeatureTail.at<double>(idx, 0) = (double)readingFloat;
					if (idx < readingInt - 1) { fscanf_s(fpTracklet, ","); }
				}
				fscanf_s(fpTracklet, ")\n");

				// location in 3D
				fscanf_s(fpTracklet, "\t\t\t\tcurrentLocation3D:(%f,%f,%f)\n", &newTracklet.currentLocation3D.x, &newTracklet.currentLocation3D.y, &newTracklet.currentLocation3D.z);

				// bAssociableNewMeasurement
				for (int compCamIdx = 0; compCamIdx < NUM_CAM; compCamIdx++)
				{
					int numFlags = 0;
					fscanf_s(fpTracklet, "\t\t\t\tbAssociableNewMeasurement[%d]:%d,{", &readingInt, &numFlags);
					for (int flagIdx = 0; flagIdx < numFlags; flagIdx++)
					{
						fscanf_s(fpTracklet, "%d", &readingInt);
						bool curFlag = 0 < readingInt ? true : false;
						newTracklet.bAssociableNewMeasurement[compCamIdx].push_back(curFlag);
						if (flagIdx < numFlags - 1) { fscanf_s(fpTracklet, ","); }
					}
					fscanf_s(fpTracklet, "}\n");
				}
				fscanf_s(fpTracklet, "\t\t\t}\n");
				vecTracklet2DSet_[camIdx].tracklets.push_back(newTracklet);
#ifdef PSN_DEBUG_MODE_
				printf("\r>> Reading 2D tracklet set at cam %d: %3.1f%%", camIdx, 100.0 * (double)(trackletIdx + 1.0) / (double)numTrackletSet);
#endif
			}
#ifdef PSN_DEBUG_MODE_
			printf("\n");
#endif

			int numActiveTracklet = 0;
			vecTracklet2DSet_[camIdx].activeTracklets.clear();
			fscanf_s(fpTracklet, "\t\tactiveTracklets:%d,{", &numActiveTracklet);
			for (int trackletIdx = 0; trackletIdx < numActiveTracklet; trackletIdx++)
			{
				fscanf_s(fpTracklet, "%d", &readingInt);
				// search active tracklet
				for (std::list<stTracklet2D>::iterator trackletIter = vecTracklet2DSet_[camIdx].tracklets.begin();
					trackletIter != vecTracklet2DSet_[camIdx].tracklets.end();
					trackletIter++)
				{
					if ((*trackletIter).id != (unsigned int)readingInt) { continue; }
					vecTracklet2DSet_[camIdx].activeTracklets.push_back(&(*trackletIter));
					break;
				}
				if (trackletIdx < numActiveTracklet - 1) { fscanf_s(fpTracklet, ","); }
			}

			int numNewTracklet = 0;
			fscanf_s(fpTracklet, "}\n\t\tnewMeasurements:%d,{", &numNewTracklet);
			for (int trackletIdx = 0; trackletIdx < numNewTracklet; trackletIdx++)
			{
				fscanf_s(fpTracklet, "%d", &readingInt);
				// search active tracklet
				for (std::list<stTracklet2D>::iterator trackletIter = vecTracklet2DSet_[camIdx].tracklets.begin();
					trackletIter != vecTracklet2DSet_[camIdx].tracklets.end();
					trackletIter++)
				{
					if ((*trackletIter).id != (unsigned int)readingInt) { continue; }
					vecTracklet2DSet_[camIdx].newMeasurements.push_back(&(*trackletIter));
					break;
				}
				if (trackletIdx < numNewTracklet - 1) { fscanf_s(fpTracklet, ","); }
			}
			fscanf_s(fpTracklet, "}\n\t}\n");
		}
		fscanf_s(fpTracklet, "}\n");
		fscanf_s(fpTracklet, "nNumTotalActive2DTracklet:%d\n\n", &readingInt);
		nNumTotalActive2DTracklet_ = (unsigned int)readingInt;

		fclose(fpTracklet);

		//---------------------------------------------------------
		// 3D TRACK RELATED
		//---------------------------------------------------------
#ifdef PSN_DEBUG_MODE_
		printf(">> Reading 3D tracks: %3.1f%%", 0);
#endif
		// file open
		sprintf_s(strFilename, "%ssnapshot_3D_track.txt", strFilepath);		
		fopen_s(&fpTrack, strFilename, "r"); 
		if (NULL == fpTrack) { return false; }
		fscanf_s(fpTrack, "numCamera:%d\n", &readingInt);
		assert(NUM_CAM == readingInt);
		fscanf_s(fpTrack, "frameIndex:%d\n\n", &readingInt);
		nCurrentFrameIdx_ = (unsigned int)readingInt;

		fscanf_s(fpTrack, "bReceiveNewMeasurement:%d\n", &readingInt);
		bReceiveNewMeasurement_ = 0 < readingInt ? true : false;
		fscanf_s(fpTrack, "bInitiationPenaltyFree:%d\n", &readingInt);
		bInitiationPenaltyFree_ = 0 < readingInt ? true : false;
		fscanf_s(fpTrack, "nNewTrackID:%d\n", &readingInt);
		nNewTrackID_ = (unsigned int)readingInt;
		fscanf_s(fpTrack, "nNewTreeID:%d\n", &readingInt);
		nNewTreeID_ = (unsigned int)readingInt;

		// track
		int numTrack = 0;
		listTrack3D_.clear();
		fscanf_s(fpTrack, "listTrack3D:%d,\n{\n", &numTrack);
		for (int trackIdx = 0; trackIdx < numTrack; trackIdx++)
		{
			Track3D newTrack;
			int parentTrackID = 0, treeID = 0;
			std::deque<unsigned int> childrenTrackID;
			fscanf_s(fpTrack, "\t{\n\t\tid:%d\n", &readingInt);
			newTrack.id = (unsigned int)readingInt;

			// curTracklet2Ds
			fscanf_s(fpTrack, "\t\tcurTracklet2Ds:{");
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				fscanf_s(fpTrack, "%d", &readingInt);
				if (-1 == readingInt)
				{
					newTrack.curTracklet2Ds.set(camIdx, NULL);
				}
				else
				{
					for (std::deque<stTracklet2D*>::iterator trackletIter = vecTracklet2DSet_[camIdx].activeTracklets.begin();
						trackletIter != vecTracklet2DSet_[camIdx].activeTracklets.end();
						trackletIter++)
					{
						if ((*trackletIter)->id != (unsigned int)readingInt) { continue; }
						newTrack.curTracklet2Ds.set(camIdx, *trackletIter);
						break;
					}
				}
				if (camIdx < NUM_CAM - 1) { fscanf_s(fpTrack, ","); }
			}
			fscanf_s(fpTrack, "}\n");

			// tracklet2DIDs
			fscanf_s(fpTrack, "\t\ttrackleIDs:\n\t\t{\n");
			for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				int numTracklet = 0;
				fscanf_s(fpTrack, "\t\t\tnumTracklet:%d,{", &numTracklet);

				for (int trackletIdx = 0; trackletIdx < numTracklet; trackletIdx++)
				{
					fscanf_s(fpTrack, "%d", &readingInt);
					newTrack.tracklet2DIDs[camIdx].push_back((unsigned int)readingInt);
					if (trackletIdx < numTracklet - 1) { fscanf_s(fpTrack, ","); }
				}

				fscanf_s(fpTrack, "}\n");
			}
			fscanf_s(fpTrack, "\t\t}\n");

			// tracklet last locations
			fscanf_s(fpTrack, "\t\ttrackletLastLocation3D:{");
			for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				float tx = 0, ty = 0, tz = 0;
				fscanf_s(fpTrack, "(%f,%f,%f)", &tx, &ty, &tz);
				newTrack.trackletLastLocation3D[camIdx] = PSN_Point3D((double)tx, (double)ty, (double)tz);
			}
			fscanf_s(fpTrack, "}\n");

			fscanf_s(fpTrack, "\t\tbActive:%d\n", &readingInt);
			newTrack.bActive = 0 < readingInt ? true : false;
			fscanf_s(fpTrack, "\t\tbValid:%d\n", &readingInt);
			newTrack.bValid = 0 < readingInt ? true : false;
			fscanf_s(fpTrack, "\t\ttreeID:%d\n",  &treeID);
			fscanf_s(fpTrack, "\t\tparentTrackID:%d\n", &parentTrackID);
			for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
				trackIter != listTrack3D_.end();
				trackIter++)
			{
				if ((unsigned int)parentTrackID != (*trackIter).id) { continue; }
				newTrack.parentTrack = &(*trackIter);
				break;
			}

			// children tracks
			int numChildrenTrack = 0;
			fscanf_s(fpTrack, "\t\tchildrenTrack:%d,{", &numChildrenTrack);
			for (int childTrackIdx = 0; childTrackIdx < numChildrenTrack; childTrackIdx++)
			{
				fscanf_s(fpTrack, "%d", &readingInt);
				childrenTrackID.push_back((unsigned int)readingInt);
				if (childTrackIdx < numChildrenTrack - 1) { fscanf_s(fpTrack, ","); }
			}
			fscanf_s(fpTrack, "}\n");
			
			// temporal information
			fscanf_s(fpTrack, "\t\ttimeStart:%d\n", &readingInt);
			newTrack.timeStart = (unsigned int)readingInt;
			fscanf_s(fpTrack, "\t\ttimeEnd:%d\n", &readingInt);
			newTrack.timeEnd = (unsigned int)readingInt;
			fscanf_s(fpTrack, "\t\ttimeGeneration:%d\n", &readingInt);
			newTrack.timeGeneration = (unsigned int)readingInt;
			fscanf_s(fpTrack, "\t\tduration:%d\n", &readingInt);
			newTrack.duration = (unsigned int)readingInt;

			// reconstrcution related
			int numReconstruction = 0;
			fscanf_s(fpTrack, "\t\treconstructions:%d,\n\t\t{\n", &numReconstruction);
			std::deque<PSN_Point3D> points(newTrack.duration);
			std::deque<PSN_Point3D> smoothedPoints(newTrack.duration);
			for (int pointIdx = 0; pointIdx < numReconstruction; pointIdx++)
			{
				stReconstruction newReconstruction;
				float x, y, z, sx, sy, sz, vx, vy, vz, maxError, costReconstruction, costSmoothedPoint, costLink;
				fscanf_s(fpTrack, "\t\t\t%d,point:(%f,%f,%f),smoothedPoint:(%f,%f,%f),velocity:(%f,%f,%f),maxError:%e,costReconstruction:%e,costSmoothedPoint:%e,costLink:%e,", 
					&readingInt, &x, &y, &z, &sx, &sy, &sz, &vx, &vy, &vz, &maxError, &costReconstruction, &costSmoothedPoint, &costLink);
				newReconstruction.bIsMeasurement = 0 < readingInt ? true : false;
				newReconstruction.point = PSN_Point3D((double)x, (double)y, (double)z);
				newReconstruction.smoothedPoint = PSN_Point3D((double)sx, (double)sy, (double)sz);
				newReconstruction.velocity = PSN_Point3D((double)vx, (double)vy, (double)vz);
				newReconstruction.maxError = (double)maxError;
				newReconstruction.costLink = (double)costLink;
				newReconstruction.costReconstruction = (double)costReconstruction;
				newReconstruction.costSmoothedPoint = (double)costSmoothedPoint;

				if (pointIdx < (int)newTrack.duration)
				{
					points[pointIdx].x = x;
					points[pointIdx].y = y;
					points[pointIdx].z = z;
					smoothedPoints[pointIdx].x = sx;
					smoothedPoints[pointIdx].y = sy;
					smoothedPoints[pointIdx].z = sz;
				}

				fscanf_s(fpTrack, "tracklet2Ds:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					fscanf_s(fpTrack, "%d", &readingInt);

					if (0 > readingInt)
					{
						newReconstruction.tracklet2Ds.set(camIdx, NULL);
					}
					else
					{
						for (std::list<stTracklet2D>::iterator trackletIter = vecTracklet2DSet_[camIdx].tracklets.begin();
							trackletIter != vecTracklet2DSet_[camIdx].tracklets.end();
							trackletIter++)
						{
							if ((*trackletIter).id != (unsigned int)readingInt) { continue; }
							newReconstruction.tracklet2Ds.set(camIdx, &(*trackletIter));
							break;
						}
					}
					if (camIdx < NUM_CAM - 1) { fscanf_s(fpTrack, ","); }
				}
				fscanf_s(fpTrack, "},");
				fscanf_s(fpTrack, "rawPoints:%d,{", &readingInt);
				for (int pointIdx = 0; pointIdx < readingInt; pointIdx++)
				{
					fscanf_s(fpTrack, "(%f,%f,%f)", &x, &y, &z);
					newReconstruction.rawPoints.push_back(PSN_Point3D((double)x, (double)y, (double)z));
					if (pointIdx < readingInt - 1) { fscanf_s(fpTrack, ","); }
				}
				fscanf_s(fpTrack, "}\n");

				newTrack.reconstructions.push_back(newReconstruction);
			}
			fscanf_s(fpTrack, "\t\t}\n");
						
			// point smoother
			int span, degree;
			fscanf_s(fpTrack, "\t\tsmoother:{span:%d,degree:%d}\n", &span, &degree);
			newTrack.smoother.SetSmoother(points, smoothedPoints, span, degree);
			newTrack.smoother.SetQsets(&precomputedQsets);

			// cost
			fscanf_s(fpTrack, "\t\tcostTotal:%e\n", &readingFloat);
			newTrack.costTotal = (double)readingFloat;
			fscanf_s(fpTrack, "\t\tcostReconstruction:%e\n", &readingFloat);
			newTrack.costReconstruction = (double)readingFloat;
			fscanf_s(fpTrack, "\t\tcostLink:%e\n", &readingFloat);
			newTrack.costLink = (double)readingFloat;
			fscanf_s(fpTrack, "\t\tcostEnter:%e\n", &readingFloat);
			newTrack.costEnter = (double)readingFloat;
			fscanf_s(fpTrack, "\t\tcostExit:%e\n", &readingFloat);
			newTrack.costExit = (double)readingFloat;
			fscanf_s(fpTrack, "\t\tcostRGB:%e\n", &readingFloat);
			newTrack.costRGB = (double)readingFloat;			

			// loglikelihood
			fscanf_s(fpTrack, "\t\tloglikelihood:%e\n", &readingFloat);
			newTrack.loglikelihood = (double)readingFloat;

			// GTP
			fscanf_s(fpTrack, "\t\tGTProb:%e\n", &readingFloat);
			newTrack.GTProb = (double)readingFloat;
			fscanf_s(fpTrack, "\t\tBranchGTProb:%e\n", &readingFloat);
			newTrack.BranchGTProb = (double)readingFloat;
			fscanf_s(fpTrack, "\t\tbWasBestSolution:%d\n", &readingInt);
			newTrack.bWasBestSolution = 0 < readingInt ? true: false;
			fscanf_s(fpTrack, "\t\tbCurrentBestSolution:%d\n", &readingInt);
			newTrack.bCurrentBestSolution = 0 < readingInt ? true : false;

			// HO-MHT
			fscanf_s(fpTrack, "\t\tbNewTrack:%d\n", &readingInt);
			newTrack.bNewTrack = 0 < readingInt ? true : false;

			// appearance
			fscanf_s(fpTrack, "\t\tlastRGBFeature:\n\t\t{\n");
			for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				int dimFeature = 0;
				fscanf_s(fpTrack, "\t\t\tlastRGBFeature[%d]:%d,(", &readingInt, &dimFeature);
				newTrack.lastRGBFeature[camIdx] = cv::Mat::zeros(dimFeature, 1, CV_64FC1);
				for (int idx = 0; idx < dimFeature; idx++)
				{
					fscanf_s(fpTrack, "%f", &readingFloat);
					newTrack.lastRGBFeature[camIdx].at<double>(idx, 0) = (double)readingFloat;
					if (idx < dimFeature - 1) { fscanf_s(fpTrack, ","); }
				}
				fscanf_s(fpTrack, ")\n");
			}
			fscanf_s(fpTrack, "\t\t}\n\t\ttimeTrackletEnded:(");
			for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			{
				fscanf_s(fpTrack, "%d", &readingInt);
				newTrack.timeTrackletEnded[camIdx] = (unsigned int)readingInt;
				if (camIdx < NUM_CAM - 1) { fscanf_s(fpTrack, ","); }				
			}
			fscanf_s(fpTrack, ")\n\t}\n");

			// generate instance
			listTrack3D_.push_back(newTrack);
			treeIDPair.push_back(std::make_pair(&listTrack3D_.back(), (unsigned int)treeID));
			childrenTrackIDPair.push_back(std::make_pair(&listTrack3D_.back(), childrenTrackID));

#ifdef PSN_DEBUG_MODE_
			printf("\r>> Reading 3D tracks: %3.1f%%", 100.0 * (double)(trackIdx + 1.0) / (double)numTrack);
#endif
		}
		fscanf_s(fpTrack, "}\n");
#ifdef PSN_DEBUG_MODE_
		printf("\n");
#endif
		
		// children track
		for (int queueIdx = 0; queueIdx < childrenTrackIDPair.size(); queueIdx++)
		{
			Track3D *curTrack = childrenTrackIDPair[queueIdx].first;
			for (int idIdx = 0; idIdx < childrenTrackIDPair[queueIdx].second.size(); idIdx++)
			{
				unsigned int curChildID = childrenTrackIDPair[queueIdx].second[idIdx];
				for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
				trackIter != listTrack3D_.end();
				trackIter++)
				{
					if (curChildID != (*trackIter).id) { continue; }
					curTrack->childrenTrack.push_back(&(*trackIter));
					break;
				}
			}
		}

#ifdef PSN_DEBUG_MODE_
		printf(">> Reading 3D track trees: %3.1f%%", 0);
#endif
		// track tree
		int numTree = 0;
		listTrackTree_.clear();
		fscanf_s(fpTrack, "listTrackTree:%d,\n{\n", &numTree);
		for (int treeIdx = 0; treeIdx < numTree; treeIdx++)
		{
			TrackTree newTree;

			fscanf_s(fpTrack, "\t{\n\t\tid:%d\n", &readingInt);
			newTree.id = (unsigned int)readingInt;
			fscanf_s(fpTrack, "\t\ttimeGeneration:%d\n", &readingInt);
			newTree.timeGeneration = (unsigned int)readingInt;
			fscanf_s(fpTrack, "\t\tbValid:%d\n", &readingInt);
			newTree.bValid = 0 < readingInt ? true : false;
			int numTracks = 0;
			fscanf_s(fpTrack, "\t\ttracks:%d,{", &numTracks);
			for (int trackIdx = 0; trackIdx < numTracks; trackIdx++)
			{
				fscanf_s(fpTrack, "%d", &readingInt);
				for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
					trackIter != listTrack3D_.end();
					trackIter++)
				{
					if ((unsigned int)readingInt != (*trackIter).id) { continue; }
					newTree.tracks.push_back(&(*trackIter));
					break;
				}
				if (trackIdx < numTracks - 1) { fscanf_s(fpTrack, ","); }
			}
			fscanf_s(fpTrack, "}\n");
			//fscanf_s(fpTrack, "\t\tnumMeasurements:%d\n", &readingInt);
			//newTree.numMeasurements = (unsigned int)readingInt;
			//// tracklet2Ds
			//int numTracklet = 0;
			//fscanf_s(fpTrack, "\t\ttrackle2Ds:\n\t\t{\n");
			//for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
			//{
			//	fscanf_s(fpTrack, "\t\t\ttrackletInfo:%d,", &numTracklet);
			//	if (0 == numTracklet)
			//	{
			//		fscanf_s(fpTrack, "{}\n");
			//		continue;
			//	}
			//	fscanf_s(fpTrack, "\n\t\t\t{\n");
			//	for (int trackletIdx = 0; trackletIdx < numTracklet; trackletIdx++)
			//	{
			//		int numRelatedTracks = 0;
			//		stTracklet2DInfo newTrackletInfo;
			//		fscanf_s(fpTrack, "\t\t\t\tid:%d,relatedTracks:%d,{", &readingInt, &numRelatedTracks);
			//		// find tracklet
			//		for (std::list<stTracklet2D>::iterator trackletIter = vecTracklet2DSet_[camIdx].tracklets.begin();
			//			trackletIter != vecTracklet2DSet_[camIdx].tracklets.end();
			//			trackletIter++)
			//		{
			//			if ((unsigned int)readingInt != (*trackletIter).id) { continue; }
			//			newTrackletInfo.tracklet2D = &(*trackletIter);
			//			break;
			//		}

			//		// find related tracks
			//		for (int idIdx = 0; idIdx < numRelatedTracks; idIdx++)
			//		{
			//			fscanf_s(fpTrack, "%d", &readingInt);
			//			for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
			//				trackIter != listTrack3D_.end();
			//				trackIter++)
			//			{
			//				if ((unsigned int)readingInt != (*trackIter).id) { continue; }
			//				newTrackletInfo.queueRelatedTracks.push_back(&(*trackIter));
			//				break;
			//			}
			//			if (trackletIdx < numTracklet - 1) { fscanf_s(fpTrack, ","); }
			//		}
			//		fscanf_s(fpTrack, "}\n");
			//		newTree.tracklet2Ds[camIdx].push_back(newTrackletInfo);					
			//	}
			//	fscanf_s(fpTrack, "\t\t\t}\n");
			//}
			//fscanf_s(fpTrack, "\t\t}\n\t}\n");
			fscanf_s(fpTrack, "\t}\n");

			listTrackTree_.push_back(newTree);

#ifdef PSN_DEBUG_MODE_
			printf("\r>> Reading 3D tracks: %3.1f%%", 100.0 * (double)(treeIdx + 1.0) / (double)numTree);
#endif
		}
		fscanf_s(fpTrack, "}\n");
#ifdef PSN_DEBUG_MODE_
		printf("\n");
#endif

		// set track's tree pointers
		for (int pairIdx = 0; pairIdx < treeIDPair.size(); pairIdx++)
		{
			Track3D *curTrack = treeIDPair[pairIdx].first;
			for (std::list<TrackTree>::iterator treeIter = listTrackTree_.begin();
				treeIter != listTrackTree_.end();
				treeIter++)
			{
				if (treeIDPair[pairIdx].second != (*treeIter).id) { continue; }
				curTrack->tree = &(*treeIter);
				break;
			}
		}
		
		// new seed tracks
		int numSeedTracks = 0;
		queueNewSeedTracks_.clear();
		fscanf_s(fpTrack, "queueNewSeedTracks:%d,{", &numSeedTracks);
		for (int trackIdx = 0; trackIdx < numSeedTracks; trackIdx++)
		{
			fscanf_s(fpTrack, "%d", &readingInt);
			if (trackIdx < numSeedTracks - 1) { fscanf_s(fpTrack, ","); }

			for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
				trackIter != listTrack3D_.end();
				trackIter++)
			{
				if ((unsigned int)readingInt != (*trackIter).id) { continue; }
				queueNewSeedTracks_.push_back(&(*trackIter));
				break;
			}
		}
		fscanf_s(fpTrack, "}\n");

		// active tracks
		int numActiveTracks = 0;
		queueActiveTrack_.clear();
		fscanf_s(fpTrack, "queueActiveTrack:%d,{", &numActiveTracks);
		for (int trackIdx = 0; trackIdx < numActiveTracks; trackIdx++)
		{
			fscanf_s(fpTrack, "%d", &readingInt);
			if (trackIdx < numActiveTracks - 1) { fscanf_s(fpTrack, ","); }

			for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
				trackIter != listTrack3D_.end();
				trackIter++)
			{
				if ((unsigned int)readingInt != (*trackIter).id) { continue; }
				queueActiveTrack_.push_back(&(*trackIter));
				break;
			}
		}
		fscanf_s(fpTrack, "}\n");

		// puased tracks
		int numPuasedTracks = 0;
		queuePausedTrack_.clear();
		fscanf_s(fpTrack, "queuePausedTrack:%d,{", &numPuasedTracks);
		for (int trackIdx = 0; trackIdx < numPuasedTracks; trackIdx++)
		{
			fscanf_s(fpTrack, "%d", &readingInt);
			if (trackIdx < numPuasedTracks - 1) { fscanf_s(fpTrack, ","); }

			for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
				trackIter != listTrack3D_.end();
				trackIter++)
			{
				if ((unsigned int)readingInt != (*trackIter).id) { continue; }
				queuePausedTrack_.push_back(&(*trackIter));
				break;
			}
		}
		fscanf_s(fpTrack, "}\n");
		
		// tracks in window
		int numInWindowTracks = 0;
		queueTracksInWindow_.clear();
		fscanf_s(fpTrack, "queueTracksInWindow:%d,{", &numInWindowTracks);
		for (int trackIdx = 0; trackIdx < numInWindowTracks; trackIdx++)
		{
			fscanf_s(fpTrack, "%d", &readingInt);
			if (trackIdx < numInWindowTracks - 1) { fscanf_s(fpTrack, ","); }

			for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
				trackIter != listTrack3D_.end();
				trackIter++)
			{
				if ((unsigned int)readingInt != (*trackIter).id) { continue; }
				queueTracksInWindow_.push_back(&(*trackIter));
				break;
			}
		}
		fscanf_s(fpTrack, "}\n");

		// tracks in the best solution
		int numBSTracks = 0;
		queueTracksInBestSolution_.clear();
		fscanf_s(fpTrack, "queueTracksInBestSolution:%d,{", &numBSTracks);
		for (int trackIdx = 0; trackIdx < numBSTracks; trackIdx++)
		{
			fscanf_s(fpTrack, "%d", &readingInt);
			if (trackIdx < numBSTracks - 1) { fscanf_s(fpTrack, ","); }

			for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
				trackIter != listTrack3D_.end();
				trackIter++)
			{
				if ((unsigned int)readingInt != (*trackIter).id) { continue; }
				queueTracksInBestSolution_.push_back(&(*trackIter));
				break;
			}
		}
		fscanf_s(fpTrack, "}\n");

		// active trees
		int numActiveTrees = 0;
		queuePtActiveTrees_.clear();
		fscanf_s(fpTrack, "queuePtActiveTrees:%d,{", &numActiveTrees);
		for (int treeIdx = 0; treeIdx < numActiveTrees; treeIdx++)
		{
			fscanf_s(fpTrack, "%d", &readingInt);
			if (treeIdx < numActiveTrees - 1) { fscanf_s(fpTrack, ","); }

			for (std::list<TrackTree>::iterator treeIter = listTrackTree_.begin();
				treeIter != listTrackTree_.end();
				treeIter++)
			{
				if ((unsigned int)readingInt != (*treeIter).id) { continue; }
				queuePtActiveTrees_.push_back(&(*treeIter));
				break;
			}
		}
		fscanf_s(fpTrack, "}\n");

		// unconfirmed trees
		int numUCTrees = 0;
		queuePtUnconfirmedTrees_.clear();
		fscanf_s(fpTrack, "queuePtUnconfirmedTrees:%d,{", &numUCTrees);
		for (int treeIdx = 0; treeIdx < numUCTrees; treeIdx++)
		{
			fscanf_s(fpTrack, "%d", &readingInt);
			if (treeIdx < numUCTrees - 1) { fscanf_s(fpTrack, ","); }

			for (std::list<TrackTree>::iterator treeIter = listTrackTree_.begin();
				treeIter != listTrackTree_.end();
				treeIter++)
			{
				if ((unsigned int)readingInt != (*treeIter).id) { continue; }
				queuePtUnconfirmedTrees_.push_back(&(*treeIter));
				break;
			}
		}
		fscanf_s(fpTrack, "}\n");
		fclose(fpTrack);

		//---------------------------------------------------------
		// RESULTS
		//---------------------------------------------------------
#ifdef PSN_DEBUG_MODE_
		printf(">> Reading instant results: %3.1f%%", 0);
#endif
		// file open
		sprintf_s(strFilename, "%ssnapshot_3D_result.txt", strFilepath);		
		fopen_s(&fpResult, strFilename, "r"); 
		if (NULL == fpResult) { return false; }
		fscanf_s(fpResult, "numCamera:%d\n", &readingInt);
		assert(NUM_CAM == readingInt);
		fscanf_s(fpResult, "frameIndex:%d\n\n", &readingInt);
		nCurrentFrameIdx_ = (unsigned int)readingInt;

		// instance result
		int numTrackingResults = 0;
		queueTrackingResult_.clear();
		fscanf_s(fpResult, "queueTrackingResult:%d,\n{\n", &numTrackingResults);
		for (int resultIdx = 0; resultIdx < numTrackingResults; resultIdx++)
		{
			stTrack3DResult curResult;
			fscanf_s(fpResult, "\t{\n\t\tframeIdx:%d\n", &readingInt);
			curResult.frameIdx = (unsigned int)readingInt;
			fscanf_s(fpResult, "\t\tprocessingTime:%f\n", &readingFloat);
			curResult.processingTime = (double)readingFloat;

			// object info
			int numObjects = 0;
			fscanf_s(fpResult, "\t\tobjectInfo:%d,\n\t\t{\n", &numObjects);
			for (int objIdx = 0; objIdx < numObjects; objIdx++)
			{
				stObject3DInfo curObject;
				fscanf_s(fpResult, "\t\t\t{\n\t\t\t\tid:%d\n", &readingInt);
				curObject.id = (unsigned int)readingInt;

				// points
				int numPoints = 0;
				fscanf_s(fpResult, "\t\t\t\trecentPoints:%d,{", &numPoints);
				for (int pointIdx = 0; pointIdx < numPoints; pointIdx++)
				{
					float x, y, z;
					fscanf_s(fpResult, "(%f,%f,%f)", &x, &y, &z);
					if (pointIdx < numPoints - 1) { fscanf_s(fpResult, ","); }
					curObject.recentPoints.push_back(PSN_Point3D((double)x, (double)y, (double)z));
				}
				fscanf_s(fpResult, "}\n");

				// 2D points
				fscanf_s(fpResult, "\t\t\t\trecentPoint2Ds:\n\t\t\t\t{\n");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{					
					fscanf_s(fpResult, "\t\t\t\t\tcam%d:%d,{", &readingInt, &numPoints);
					for (int pointIdx = 0; pointIdx < numPoints; pointIdx++)
					{
						float x, y;
						fscanf_s(fpResult, "(%f,%f)", &x, &y);
						if (pointIdx < numPoints - 1) { fscanf_s(fpResult, ","); }
						curObject.recentPoint2Ds[camIdx].push_back(PSN_Point2D((double)x, (double)y));
					}
					fscanf_s(fpResult, "}\n");
				}
				fscanf_s(fpResult, "\t\t\t\t}\n");

				// 3D box points in each view
				fscanf_s(fpResult, "\t\t\t\tpoint3DBox:\n\t\t\t\t{\n");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{					
					fscanf_s(fpResult, "\t\t\t\t\tcam%d:%d,{", &readingInt, &numPoints);
					for (int pointIdx = 0; pointIdx < numPoints; pointIdx++)
					{
						float x, y;
						fscanf_s(fpResult, "(%f,%f)", &x, &y);
						if (pointIdx < numPoints - 1) { fscanf_s(fpResult, ","); }
						curObject.point3DBox[camIdx].push_back(PSN_Point2D((double)x, (double)y));
					}
					fscanf_s(fpResult, "}\n");
				}
				fscanf_s(fpResult, "\t\t\t\t}\n");

				// rects
				fscanf_s(fpResult, "\t\t\t\trectInViews:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					float x, y, w, h;
					fscanf_s(fpResult, "(%f,%f,%f,%f)", &x, &y, &w, &h);
					if (camIdx < NUM_CAM - 1) { fscanf_s(fpResult, ","); }
					curObject.rectInViews[camIdx] = PSN_Rect((double)x, (double)y, (double)w, (double)h);
				}
				fscanf_s(fpResult, "}\n");

				// visibility
				fscanf_s(fpResult, "\t\t\t\tbVisibleInViews:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					fscanf_s(fpResult, "%d", &readingInt);
					if (camIdx < NUM_CAM - 1) { fscanf_s(fpResult, ","); }
					curObject.bVisibleInViews[camIdx] = 0 < readingInt ? true : false;
				}
				fscanf_s(fpResult, "}\n\t\t\t}\n");

				curResult.object3DInfo.push_back(curObject);
			}
			fscanf_s(fpResult, "\t\t}\n\t}\n");

			queueTrackingResult_.push_back(curResult);

#ifdef PSN_DEBUG_MODE_
			printf("\r>> Reading instant results: %3.1f%%", 100.0 * (double)(resultIdx + 1.0) / (double)numTrackingResults);
#endif
		}
		fscanf_s(fpResult, "}\n");
#ifdef PSN_DEBUG_MODE_
		printf("\n>> Reading deferred results: %3.1f%%", 0);
#endif

		// deferred result
		numTrackingResults = 0;
		queueDeferredTrackingResult_.clear();
		fscanf_s(fpResult, "queueDeferredTrackingResult:%d,\n{\n", &numTrackingResults);
		for (int resultIdx = 0; resultIdx < numTrackingResults; resultIdx++)
		{
			stTrack3DResult curResult;
			fscanf_s(fpResult, "\t{\n\t\tframeIdx:%d\n", &readingInt);
			curResult.frameIdx = (unsigned int)readingInt;
			fscanf_s(fpResult, "\t\tprocessingTime:%f\n", &readingFloat);
			curResult.processingTime = (double)readingFloat;

			// object info
			int numObjects = 0;
			fscanf_s(fpResult, "\t\tobjectInfo:%d,\n\t\t{\n", &numObjects);
			for (int objIdx = 0; objIdx < numObjects; objIdx++)
			{
				stObject3DInfo curObject;
				fscanf_s(fpResult, "\t\t\t{\n\t\t\t\tid:%d\n", &readingInt);
				curObject.id = (unsigned int)readingInt;

				// points
				int numPoints = 0;
				fscanf_s(fpResult, "\t\t\t\trecentPoints:%d,{", &numPoints);
				for (int pointIdx = 0; pointIdx < numPoints; pointIdx++)
				{
					float x, y, z;
					fscanf_s(fpResult, "(%f,%f,%f)", &x, &y, &z);
					if (pointIdx < numPoints - 1) { fscanf_s(fpResult, ","); }
					curObject.recentPoints.push_back(PSN_Point3D((double)x, (double)y, (double)z));
				}
				fscanf_s(fpResult, "}\n");

				// 2D points
				fscanf_s(fpResult, "\t\t\t\trecentPoint2Ds:\n\t\t\t\t{\n");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{					
					fscanf_s(fpResult, "\t\t\t\t\tcam%d:%d,{", &readingInt, &numPoints);
					for (int pointIdx = 0; pointIdx < numPoints; pointIdx++)
					{
						float x, y;
						fscanf_s(fpResult, "(%f,%f)", &x, &y);
						if (pointIdx < numPoints - 1) { fscanf_s(fpResult, ","); }
						curObject.recentPoint2Ds[camIdx].push_back(PSN_Point2D((double)x, (double)y));
					}
					fscanf_s(fpResult, "}\n");
				}
				fscanf_s(fpResult, "\t\t\t\t}\n");

				// 3D box points in each view
				fscanf_s(fpResult, "\t\t\t\tpoint3DBox:\n\t\t\t\t{\n");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{					
					fscanf_s(fpResult, "\t\t\t\t\tcam%d:%d,{", &readingInt, &numPoints);
					for (int pointIdx = 0; pointIdx < numPoints; pointIdx++)
					{
						float x, y;
						fscanf_s(fpResult, "(%f,%f)", &x, &y);
						if (pointIdx < numPoints - 1) { fscanf_s(fpResult, ","); }
						curObject.point3DBox[camIdx].push_back(PSN_Point2D((double)x, (double)y));
					}
					fscanf_s(fpResult, "}\n");
				}
				fscanf_s(fpResult, "\t\t\t\t}\n");

				// rects
				fscanf_s(fpResult, "\t\t\t\trectInViews:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					float x, y, w, h;
					fscanf_s(fpResult, "(%f,%f,%f,%f)", &x, &y, &w, &h);
					if (camIdx < NUM_CAM - 1) { fscanf_s(fpResult, ","); }
					curObject.rectInViews[camIdx] = PSN_Rect((double)x, (double)y, (double)w, (double)h);
				}
				fscanf_s(fpResult, "}\n");

				// visibility
				fscanf_s(fpResult, "\t\t\t\tbVisibleInViews:{");
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
				{
					fscanf_s(fpResult, "%d", &readingInt);
					if (camIdx < NUM_CAM - 1) { fscanf_s(fpResult, ","); }
					curObject.bVisibleInViews[camIdx] = 0 < readingInt ? true : false;
				}
				fscanf_s(fpResult, "}\n\t\t\t}\n");

				curResult.object3DInfo.push_back(curObject);
			}
			fscanf_s(fpResult, "\t\t}\n\t}\n");

			queueDeferredTrackingResult_.push_back(curResult);

#ifdef PSN_DEBUG_MODE_
			printf("\r>> Reading deferred results: %3.1f%%", 100.0 * (double)(resultIdx + 1.0) / (double)numTrackingResults);
#endif
		}
#ifdef PSN_DEBUG_MODE_
		printf("\n");
#endif
		fscanf_s(fpResult, "}\n");
		fclose(fpResult);

		//---------------------------------------------------------
		// HYPOTHESES
		//---------------------------------------------------------
		// file open
		sprintf_s(strFilename, "%ssnapshot_3D_hypotheses.txt", strFilepath);		
		fopen_s(&fpHypothesis, strFilename, "r"); 
		if (NULL == fpHypothesis) { return false; }
		fscanf_s(fpHypothesis, "numCamera:%d\n", &readingInt);
		assert(NUM_CAM == readingInt);
		fscanf_s(fpHypothesis, "frameIndex:%d\n\n", &readingInt);
		nCurrentFrameIdx_ = (unsigned int)readingInt;

#ifdef PSN_DEBUG_MODE_
		printf(">> Reading previous hypotheses: %3.1f%%", 0);
#endif
		// queuePrevGlobalHypotheses
		int numPrevGH = 0;
		queuePrevGlobalHypotheses_.clear();
		fscanf_s(fpHypothesis, "queuePrevGlobalHypotheses:%d,\n{\n", &numPrevGH);
		for (int hypothesisIdx = 0; hypothesisIdx < numPrevGH; hypothesisIdx++)
		{
			stGlobalHypothesis newHypothesis;

			int numSelectedTracks = 0;
			fscanf_s(fpHypothesis, "\t{\n\t\tselectedTracks:%d,{", &numSelectedTracks);			
			for (int trackIdx = 0; trackIdx < numSelectedTracks; trackIdx++)
			{
				fscanf_s(fpHypothesis, "%d", &readingInt);
				if (trackIdx < numSelectedTracks - 1) { fscanf_s(fpHypothesis, ","); }

				for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
					trackIter != listTrack3D_.end();
					trackIter++)
				{
					if ((unsigned int)readingInt != (*trackIter).id) { continue; }
					newHypothesis.selectedTracks.push_back(&(*trackIter));
					break;
				}
			}
			fscanf_s(fpHypothesis, "}\n");

			int numRelatedTracks = 0;
			fscanf_s(fpHypothesis, "\t\trelatedTracks:%d,{", &numRelatedTracks);			
			for (int trackIdx = 0; trackIdx < numRelatedTracks; trackIdx++)
			{
				fscanf_s(fpHypothesis, "%d", &readingInt);
				if (trackIdx < numRelatedTracks - 1) { fscanf_s(fpHypothesis, ","); }

				for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
					trackIter != listTrack3D_.end();
					trackIter++)
				{
					if ((unsigned int)readingInt != (*trackIter).id) { continue; }
					newHypothesis.relatedTracks.push_back(&(*trackIter));
					break;
				}
			}
			fscanf_s(fpHypothesis, "}\n");
			fscanf_s(fpHypothesis, "\t\tlogLikelihood:%e\n", &readingFloat);
			newHypothesis.logLikelihood = (double)readingFloat;
			fscanf_s(fpHypothesis, "\t\tprobability:%e\n", &readingFloat);
			newHypothesis.probability = (double)readingFloat;
			fscanf_s(fpHypothesis, "\t\tbValid:%d\n\t}\n", &readingInt);
			newHypothesis.bValid = 0 < readingInt ? true : false;

			queuePrevGlobalHypotheses_.push_back(newHypothesis);

#ifdef PSN_DEBUG_MODE_
			printf("\r>> Reading previous hypotheses: %3.1f%%", 100.0 * (double)(hypothesisIdx + 1.0) / (double)numPrevGH);
#endif
		}
		fscanf_s(fpHypothesis, "}\n");

#ifdef PSN_DEBUG_MODE_
		printf("\n>> Reading current hypotheses: %3.1f%%", 0);
#endif
		// queueCurrGlobalHypotheses
		int numCurrGH = 0;
		queueCurrGlobalHypotheses_.clear();
		fscanf_s(fpHypothesis, "queueCurrGlobalHypotheses:%d,\n{\n", &numCurrGH);
		for (int hypothesisIdx = 0; hypothesisIdx < numCurrGH; hypothesisIdx++)
		{
			stGlobalHypothesis newHypothesis;

			int numSelectedTracks = 0;
			fscanf_s(fpHypothesis, "\t{\n\t\tselectedTracks:%d,{", &numSelectedTracks);			
			for (int trackIdx = 0; trackIdx < numSelectedTracks; trackIdx++)
			{
				fscanf_s(fpHypothesis, "%d", &readingInt);
				if (trackIdx < numSelectedTracks - 1) { fscanf_s(fpHypothesis, ","); }

				for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
					trackIter != listTrack3D_.end();
					trackIter++)
				{
					if ((unsigned int)readingInt != (*trackIter).id) { continue; }
					newHypothesis.selectedTracks.push_back(&(*trackIter));
					break;
				}
			}
			fscanf_s(fpHypothesis, "}\n");

			int numRelatedTracks = 0;
			fscanf_s(fpHypothesis, "\t\trelatedTracks:%d,{", &numRelatedTracks);			
			for (int trackIdx = 0; trackIdx < numRelatedTracks; trackIdx++)
			{
				fscanf_s(fpHypothesis, "%d", &readingInt);
				if (trackIdx < numRelatedTracks - 1) { fscanf_s(fpHypothesis, ","); }

				for (std::list<Track3D>::iterator trackIter = listTrack3D_.begin();
					trackIter != listTrack3D_.end();
					trackIter++)
				{
					if ((unsigned int)readingInt != (*trackIter).id) { continue; }
					newHypothesis.relatedTracks.push_back(&(*trackIter));
					break;
				}
			}
			fscanf_s(fpHypothesis, "}\n");
			fscanf_s(fpHypothesis, "\t\tlogLikelihood:%e\n", &readingFloat);
			newHypothesis.logLikelihood = (double)readingFloat;
			fscanf_s(fpHypothesis, "\t\tprobability:%e\n", &readingFloat);
			newHypothesis.probability = (double)readingFloat;
			fscanf_s(fpHypothesis, "\t\tbValid:%d\n\t}\n", &readingInt);
			newHypothesis.bValid = 0 < readingInt ? true : false;

			queueCurrGlobalHypotheses_.push_back(newHypothesis);
#ifdef PSN_DEBUG_MODE_
			printf("\r>> Reading current hypotheses: %3.1f%%", 100.0 * (double)(hypothesisIdx + 1.0) / (double)numCurrGH);
#endif
		}
		fscanf_s(fpHypothesis, "}\n");
		fclose(fpHypothesis);
#ifdef PSN_DEBUG_MODE_
		printf("\n");
#endif

		//---------------------------------------------------------
		// VISUALIZATION
		//---------------------------------------------------------
		// file open
		sprintf_s(strFilename, "%ssnapshot_3D_info.txt", strFilepath);		
		fopen_s(&fpInfo, strFilename, "r"); 
		if (NULL == fpInfo) { return false; }
		fscanf_s(fpInfo, "numCamera:%d\n", &readingInt);
		assert(NUM_CAM == readingInt);
		fscanf_s(fpInfo, "frameIndex:%d\n\n", &readingInt);
		nCurrentFrameIdx_ = (unsigned int)readingInt;

		fscanf_s(fpInfo, "nNewVisualizationID:%d\n", &readingInt);
		nNewVisualizationID_ = (unsigned int)readingInt;

		int numPair = 0;
		queuePairTreeIDToVisualizationID_.clear();
		fscanf_s(fpInfo, "queuePairTreeIDToVisualizationID:%d,{", &numPair);
		for (int pairIdx = 0; pairIdx < numPair; pairIdx++)
		{
			int id1, id2;
			fscanf_s(fpInfo, "(%d,%d)", &id1, &id2);
			queuePairTreeIDToVisualizationID_.push_back(std::make_pair(id1, id2));
		}
		fscanf_s(fpInfo, "}\n");

		fclose(fpInfo);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](LoadSnapShot) cannot open file! error code %d\n", dwError);
		return false;
	}

#ifdef PSN_DEBUG_MODE_
	printf("done!\n");
#endif
	return true;
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
