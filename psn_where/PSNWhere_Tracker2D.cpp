#include "PSNWhere_Tracker2D.h"
#include <limits>
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\opencv.hpp"
#include "helpers\PSNWhere_Hungarian.h"

/////////////////////////////////////////////////////////////////////////
// PARAMETERS
/////////////////////////////////////////////////////////////////////////
#define PSN_2D_FEATURE_MIN_NUM_TRACK (4)
#define PSN_2D_FEATURE_MAX_NUM_TRACK (100)
#define PSN_2D_FEATURE_CLUSTER_RADIUS_RATIO (0.2)
#define PSN_2D_FEATURE_WIN_SIZE_RATIO (1.0)
#define PSN_2D_BACKTRACKING_INTERVAL (4)
//#define PSN_2D_OPTICALFLOW_SCALE (0.5)
#define PSN_2D_OPTICALFLOW_SCALE (1.0)

//#define PSN_2D_MAX_HEIGHT (2300)
#define PSN_2D_MAX_HEIGHT (3000)
#define PSN_2D_MIN_HEIGHT (1400)
#define PSN_2D_BOX_MAX_DISTANCE (1.0)
#define PSN_2D_MAX_DETECTION_DISTANCE (600)
#define PSN_2D_MAX_HEIGHT_DIFFERENCE (400)
#define PSN_2D_MIN_CONFIDENCE (0.01)

#define PSN_2D_COST_COEF_FEATURE (1.0f)
#define PSN_2D_COST_COEF_NSSD (1.0f)
#define PSN_2D_COST_THRESHOLD_FEATURE (0.7f)
#define PSN_2D_COST_THRESHOLD_NSSD (30.0f)

#ifdef PSN_2D_DEBUG_DISPLAY_
	#define PSN_2D_DEBUG_DISPLAY_RECORD_
#endif
#define PSN_2D_DEBUG_DISPLAY_SCALE (1.0f)

// predefined
const float PSN_COST_MAX = PSN_P_INF_F * 0.5;
const double PSN_2D_OPTICALFLOW_SCALE_RECOVER = 1 / PSN_2D_OPTICALFLOW_SCALE;

/////////////////////////////////////////////////////////////////////////
// LOCAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////
// MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////

/************************************************************************
 Method Name: CPSNWhere_Tracker2D
 Description: 
	- 클래스 생성자
 Input Arguments:
	-
	-
 Return Values:
	- class instance
************************************************************************/
CPSNWhere_Tracker2D::CPSNWhere_Tracker2D(void)
	: m_bInit(false)
	, m_nCamID(0)
{
#ifdef  PSN_2D_DEBUG_DISPLAY_
	this->m_bOutputVideoInit = false;
#endif
	this->m_vecPtGrayFrameBuffer.resize(PSN_2D_BACKTRACKING_INTERVAL, NULL);
}


/************************************************************************
 Method Name: ~CPSNWhere_Tracker2D
 Description: 
	- 클래스 소멸자
 Input Arguments:
	-
	-
 Return Values:
	- none
************************************************************************/
CPSNWhere_Tracker2D::~CPSNWhere_Tracker2D(void)
{
}


/************************************************************************
 Method Name: Initialize
 Description: 
	- 클래스 초기화
 Input Arguments:
	-
	-
 Return Values:
	- class instance
************************************************************************/
void CPSNWhere_Tracker2D::Initialize(unsigned int nCamID, stCalibrationInfo *stCalibInfo)
{
	if(this->m_bInit)
	{
		printf("[WARNING] class \"CPSNWhere_Tracker2D\" is already initialized\n");
		return;
	}
	
	this->m_bSnapshotLoaded = false;
	this->m_nCamID = nCamID;	
	this->m_nCurrentFrameIdx = 0;
	this->m_stTrack2DResult.camID = m_nCamID;
	this->m_stTrack2DResult.frameIdx = 0;
	this->m_stTrack2DResult.object2DInfos.clear();

	// calibration related
	this->m_stCalibrationInfos.cCamModel = stCalibInfo->cCamModel;
	this->m_stCalibrationInfos.matProjectionSensitivity = stCalibInfo->matProjectionSensitivity.clone();
	this->m_stCalibrationInfos.matDistanceFromBoundary = stCalibInfo->matDistanceFromBoundary.clone();
	this->m_nInputWidth = this->m_stCalibrationInfos.cCamModel.width();
	this->m_nInputHeight = this->m_stCalibrationInfos.cCamModel.height();

	// detection related
	this->m_vecDetection2D.clear();

	// tracker related
	this->m_nNewTrackerID = 0;
	this->m_listTracker2D.clear();
	this->m_queueActiveTracker2D.clear();

	// input related
	for(std::vector<cv::Mat*>::iterator bufferIter = this->m_vecPtGrayFrameBuffer.begin();
		bufferIter != this->m_vecPtGrayFrameBuffer.end();
		bufferIter++)
	{
		if(NULL == *bufferIter){ continue; }
		delete *bufferIter;
		*bufferIter = NULL;
	}
	this->m_vecPtGrayFrameBuffer.clear();
	this->m_vecPtGrayFrameBuffer.resize(PSN_2D_BACKTRACKING_INTERVAL, NULL);

	// feature tracking related
	this->m_detector = cv::FeatureDetector::create("GridFAST");
	this->m_matMaskForFeature = cv::Mat((int)((double)this->m_nInputHeight * PSN_2D_OPTICALFLOW_SCALE), (int)((double)this->m_nInputWidth * PSN_2D_OPTICALFLOW_SCALE), CV_8UC1);
	this->m_matMaskForFeature = cv::Scalar(0);
	this->m_termCriteria = cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);

	//this->m_matMaskForPatch = cv::imread("helpers/PedTemplate.png");
	//cv::cvtColor(this->m_matMaskForPatch, this->m_matMaskForPatch, CV_RGB2GRAY);
	//this->m_matMaskForPatch.convertTo(this->m_matMaskForPatch, CV_32FC1);
	//this->m_matMaskForPatch = (1.0f/255.0f) * this->m_matMaskForPatch;

	// initialization flag
	this->m_bInit = true;

#ifdef PSN_2D_DEBUG_DISPLAY_
	//-----------------------
	// DEBUG
	//-----------------------
	sprintf_s(this->m_strWindowName, "Result cam%d", nCamID);
	cv::namedWindow(this->m_strWindowName);
	this->m_vecColors = psn::GenerateColors(400);
#endif
}


/************************************************************************
 Method Name: Finalize
 Description: 
	- 클래스 종료 절차
 Input Arguments:
	-
	-
 Return Values:
	- class instance
************************************************************************/
void CPSNWhere_Tracker2D::Finalize(void)
{
	if(!this->m_bInit)
	{
		printf("[WARNING] class \"CPSNWhere_Tracker2D\" is already finalized\n");
		return;
	}

#ifdef SAVE_SNAPSHOT_
	this->SaveSnapshot(SNAPSHOT_PATH);
#endif

	this->m_stTrack2DResult.object2DInfos.clear();

	// calibration related
	this->m_stCalibrationInfos.matProjectionSensitivity.release();
	this->m_stCalibrationInfos.matDistanceFromBoundary.release();

	// detection related
	this->m_vecDetection2D.clear();

	// tracker related
	for(std::list<stTracker2D>::iterator trackerIter = this->m_listTracker2D.begin();
		trackerIter != this->m_listTracker2D.end();
		trackerIter++)
	{
		(*trackerIter).boxes.clear();
		(*trackerIter).featurePoints.clear();
		(*trackerIter).trackedPoints.clear();
	}
	this->m_listTracker2D.clear();
	this->m_queueActiveTracker2D.clear();
	//this->m_matMaskForPatch.release();

	// input related
	for(std::vector<cv::Mat*>::iterator bufferIter = this->m_vecPtGrayFrameBuffer.begin();
		bufferIter != this->m_vecPtGrayFrameBuffer.end();
		bufferIter++)
	{
		if(NULL == *bufferIter){ continue; }
		delete *bufferIter;
		*bufferIter = NULL;
	}
	
	// initialization flag
	this->m_bInit = false;

#ifdef PSN_2D_DEBUG_DISPLAY_
	//-----------------------
	// DEBUG
	//-----------------------
#ifdef PSN_2D_DEBUG_DISPLAY_RECORD_
	if(this->m_bOutputVideoInit)
	{
		cvReleaseVideoWriter(&this->m_vwOutputVideo);
		this->m_bOutputVideoInit = false;
	}
#endif
	cv::destroyWindow(this->m_strWindowName);
#endif
}


/************************************************************************
 Method Name: Run
 Description: 
	- 2D tracklet generation
 Input Arguments:
	- curDetectionResult: detection result of current frame
	- curFrame: current input frame image
	- frameIdx: current frame index
 Return Values:
	- stTrack2DResult: 
************************************************************************/
stTrack2DResult& CPSNWhere_Tracker2D::Run(std::vector<stDetection> curDetectionResult, cv::Mat *curFrame, unsigned int frameIdx)
{
	assert(this->m_bInit);

#ifdef PSN_DEBUG_MODE_
	printf("[CPSNWhere_Tracker2D](Run) start with frame number %d\n", frameIdx);

	//-----------------------------------------------
	// SPEED TEST
	clock_t timer_start = clock();
	//-----------------------------------------------
#endif

	// insert to buffer
	cv::Mat grayImage;
	cv::cvtColor(*curFrame, grayImage, CV_BGR2GRAY);	
	if(NULL == this->m_vecPtGrayFrameBuffer.back())
	{
		this->m_vecPtGrayFrameBuffer.back() = new cv::Mat((int)((double)curFrame->rows * PSN_2D_OPTICALFLOW_SCALE), (int)((double)curFrame->cols * PSN_2D_OPTICALFLOW_SCALE), CV_8UC1);
	}
	cv::resize(grayImage, *this->m_vecPtGrayFrameBuffer.back(), cv::Size((int)((double)curFrame->cols * PSN_2D_OPTICALFLOW_SCALE), (int)((double)curFrame->rows * PSN_2D_OPTICALFLOW_SCALE)));	
	grayImage.release();

#ifdef LOAD_SNAPSHOT_
	if (!this->m_bSnapshotLoaded) { this->m_bSnapshotLoaded = this->LoadSnapshot(SNAPSHOT_PATH); }
	if (this->m_bSnapshotLoaded && frameIdx <= this->m_nCurrentFrameIdx)
	{
		// buffer circulation
		std::vector<cv::Mat*> vecNewPtGrayFrameBuffer;	
		vecNewPtGrayFrameBuffer.reserve(PSN_2D_BACKTRACKING_INTERVAL);
		vecNewPtGrayFrameBuffer.assign(this->m_vecPtGrayFrameBuffer.begin() + 1, this->m_vecPtGrayFrameBuffer.end());
		vecNewPtGrayFrameBuffer.push_back(*this->m_vecPtGrayFrameBuffer.begin());
		this->m_vecPtGrayFrameBuffer = vecNewPtGrayFrameBuffer;
		vecNewPtGrayFrameBuffer.clear();

		return this->m_stTrack2DResult;
	}
#endif
#ifdef PSN_2D_DEBUG_DISPLAY_
	//-----------------------
	// DEBUG
	//-----------------------
	this->m_matDebugDisplay = curFrame->clone();
	//cv::resize(testMat, testMat, cv::Size(testMat.size().width * PSN_2D_DEBUG_DISPLAY_SCALE, testMat.size().height * PSN_2D_DEBUG_DISPLAY_SCALE));
#endif

	this->m_nCurrentFrameIdx = frameIdx;
	this->m_stTrack2DResult.frameIdx = frameIdx;

	/////////////////////////////////////////////////////////////////////////////
	// BACKWARD FEATURE TRACKING
	/////////////////////////////////////////////////////////////////////////////	
	size_t numDetection = this->Track2D_BackwardFeatureTracking(curDetectionResult);
	
	/////////////////////////////////////////////////////////////////////////////
	// FORWARD FEATURE TRACKING
	/////////////////////////////////////////////////////////////////////////////
	std::vector<float> matchingCostArray = this->Track2D_ForwardTrackingAndGetMatchingScore();

	/////////////////////////////////////////////////////////////////////////////
	// BIPARTITE MATHCING & TRACKER UPDATE
	/////////////////////////////////////////////////////////////////////////////
	this->Track2D_MatchingAndUpdating(matchingCostArray);
	
	/////////////////////////////////////////////////////////////////////////////
	// WRAP-UP
	/////////////////////////////////////////////////////////////////////////////

	// buffer circulation
	std::vector<cv::Mat*> vecNewPtGrayFrameBuffer;	
	vecNewPtGrayFrameBuffer.reserve(PSN_2D_BACKTRACKING_INTERVAL);
	vecNewPtGrayFrameBuffer.assign(this->m_vecPtGrayFrameBuffer.begin() + 1, this->m_vecPtGrayFrameBuffer.end());
	vecNewPtGrayFrameBuffer.push_back(*this->m_vecPtGrayFrameBuffer.begin());
	this->m_vecPtGrayFrameBuffer = vecNewPtGrayFrameBuffer;
	vecNewPtGrayFrameBuffer.clear();

#ifdef PSN_DEBUG_MODE_
	//-----------------------------------------------
	// SPEED TEST
	clock_t timer_end = clock();
	double elapsedTime;
	elapsedTime = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
	printf("processing time of GRID FAST detector: %f\n", elapsedTime);
	//-----------------------------------------------
	
	// file save
	//this->FilePrintTracklet();

#ifdef PSN_2D_DEBUG_DISPLAY_
	//-----------------------
	// DEBUG
	//-----------------------

	// writing frame info
	char strFrameInfo[100];
	sprintf_s(strFrameInfo, "Frame: %04d", frameIdx);
	cv::rectangle(this->m_matDebugDisplay, cv::Rect(5, 2, 145, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(this->m_matDebugDisplay, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));

	// writing processing time
	sprintf_s(strFrameInfo, "processing time: %f", elapsedTime);
	cv::rectangle(this->m_matDebugDisplay, cv::Rect(5, 30, 175, 18), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(this->m_matDebugDisplay, strFrameInfo, cv::Point(6, 42), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));


	//cv::resize(testMat, testMat, cv::Size(testMat.size().width * 2, testMat.size().height * 2));
	cv::imshow(this->m_strWindowName, this->m_matDebugDisplay);
	cv::waitKey(1);
	//cv::waitKey(0);

#ifdef PSN_2D_DEBUG_DISPLAY_RECORD_
	// record related	
	IplImage *currentFrame = new IplImage(this->m_matDebugDisplay);
	if(!this->m_bOutputVideoInit)
	{
		time_t curTimer = time(NULL);
		struct tm timeStruct;
		localtime_s(&timeStruct, &curTimer);
		char resultFileDate[256];
		char resultOutputFileName[256];
		sprintf_s(resultFileDate, "%sresult_2DTracking_%02d%02d%02d_%02d%02d%02d_cam%02d", 
			RESULT_SAVE_PATH,
			timeStruct.tm_year + 1900, 
			timeStruct.tm_mon+1,
			timeStruct.tm_mday, 
			timeStruct.tm_hour, 
			timeStruct.tm_min, 
			timeStruct.tm_sec,
			this->m_nCamID);

		sprintf_s(resultOutputFileName, "%s.avi", resultFileDate);
		this->m_vwOutputVideo = cvCreateVideoWriter(resultOutputFileName, CV_FOURCC('M','J','P','G'), 15, cvGetSize(currentFrame), 1);
		this->m_bOutputVideoInit = true;
	}	
	cvWriteFrame(this->m_vwOutputVideo, currentFrame);
	delete currentFrame;
#endif
#endif
	this->m_matDebugDisplay.release();
#endif

	return this->m_stTrack2DResult;
}


/************************************************************************
 Method Name: FindInlierFeatures
 Description: 
	- Find inlier feature points
 Input Arguments:
	- vecInputFeatures:
	- vecOutputFeatures:
	- vecPointStatus:
 Return Values:
	- portion of inlier
************************************************************************/
std::vector<cv::Point2f> CPSNWhere_Tracker2D::FindInlierFeatures(std::vector<cv::Point2f> *vecInputFeatures, 
																 std::vector<cv::Point2f> *vecOutputFeatures, 
																 std::vector<unsigned char> *vecPointStatus)
{
	size_t numTrackedFeatures = 0;

	// find center of disparity
	cv::Point2f disparityCenter(0.0f, 0.0f);	
	std::vector<cv::Point2f> vecDisparity;
	std::vector<size_t> vecInlierIndex;
	for(size_t pointIdx = 0; pointIdx < vecPointStatus->size(); pointIdx++)
	{
		if(!(*vecPointStatus)[pointIdx])
		{
			continue;
		}		
		vecDisparity.push_back((*vecOutputFeatures)[pointIdx] - (*vecInputFeatures)[pointIdx]);
		disparityCenter += vecDisparity.back();
		vecInlierIndex.push_back(pointIdx);
		numTrackedFeatures++;
	}
	disparityCenter = (1/(float)numTrackedFeatures) * disparityCenter;

	// find distribution of disparity norm
	float norm;
	float normAverage = 0.0f;
	float normSqauredAverage = 0.0f;
	float normStd = 0.0;
	std::vector<float> vecNorm;
	for(size_t pointIdx = 0; pointIdx < vecDisparity.size(); pointIdx++)
	{
		norm = (float)cv::norm(vecDisparity[pointIdx] - disparityCenter);
		vecNorm.push_back(norm);
		normAverage += norm;
		normSqauredAverage += norm * norm;
	}
	normAverage /= (float)numTrackedFeatures;
	normSqauredAverage /= (float)numTrackedFeatures;
	normStd = sqrtf(((float)numTrackedFeatures/((float)numTrackedFeatures - 1)) * (normSqauredAverage - (normAverage * normAverage)));
	
	std::vector<cv::Point2f> vecInlierFeatures;
	for(size_t pointIdx = 0; pointIdx < vecNorm.size(); pointIdx++)
	{
		if(abs(vecNorm[pointIdx] - normAverage) > 1 * normStd)
		{
			continue;
		}
		vecInlierFeatures.push_back((*vecOutputFeatures)[vecInlierIndex[pointIdx]]);
	}

	return vecInlierFeatures;
}


/************************************************************************
 Method Name: LocalSearchKLT
 Description: 
	- estimate current box location with feature tracking result
 Input Arguments:
	- preFeatures: feature positions at the previous frame
	- curFeatures: feature positions at the current frame
	- featureStatus(output): indicates which feature is inlier
 Return Values:
	- PSN_Rect: estimated box
************************************************************************/
#define PSN_LOCAL_SEARCH_KLT_PORTION_INLIER (false)
#define PSN_LOCAL_SEARCH_KLT_MINIMUM_MOVEMENT (0.1)
#define PSN_LOCAL_SEARCH_KLT_NEIGHBOR_WINDOW_SIZE_RATIO (0.2)
PSN_Rect CPSNWhere_Tracker2D::LocalSearchKLT(
	PSN_Rect preBox, 
	cv::vector<cv::Point2f> &preFeatures, 
	cv::vector<cv::Point2f> &curFeatures, 
	cv::vector<size_t> &inlierFeatureIndex)
{
	size_t numFeatures = preFeatures.size();
	size_t numMovingFeatures = 0;
	inlierFeatureIndex.clear();
	inlierFeatureIndex.reserve(numFeatures);

	// find disparity of moving features
	std::vector<PSN_Point2D> vecMovingVector;
	std::vector<size_t> vecMovingFeatuerIdx;
	std::vector<double> vecDx;
	std::vector<double> vecDy;
	vecMovingVector.reserve(numFeatures);
	vecMovingFeatuerIdx.reserve(numFeatures);
	vecDx.reserve(numFeatures);		
	vecDy.reserve(numFeatures);		
	PSN_Point2D movingVector;
	double disparity = 0.0;
	for(size_t featureIdx = 0; featureIdx < numFeatures; featureIdx++)
	{
		movingVector = curFeatures[featureIdx] - preFeatures[featureIdx];
		disparity = movingVector.norm_L2();
		if(disparity < PSN_LOCAL_SEARCH_KLT_MINIMUM_MOVEMENT * PSN_2D_OPTICALFLOW_SCALE)
		{
			continue;
		}
		vecMovingVector.push_back(movingVector);
		vecMovingFeatuerIdx.push_back(featureIdx);
		vecDx.push_back(movingVector.x);
		vecDy.push_back(movingVector.y);
		numMovingFeatures++;
	}

	// check static movement
	if(numMovingFeatures < numFeatures * 0.5)
	{
		return preBox;
	}

	std::sort(vecDx.begin(), vecDx.end());
	std::sort(vecDy.begin(), vecDy.end());

	// estimate major disparity
	double windowSize = preBox.w * PSN_LOCAL_SEARCH_KLT_NEIGHBOR_WINDOW_SIZE_RATIO * PSN_2D_OPTICALFLOW_SCALE;
	size_t maxNeighborX = 0, maxNeighborY = 0;
	PSN_Point2D estimatedDisparity;	
	for(size_t disparityIdx = 0; disparityIdx < numMovingFeatures; disparityIdx++)
	{
		size_t numNeighborX = 0;
		size_t numNeighborY = 0;
		for(size_t compIdx = 0; compIdx < numMovingFeatures; compIdx++)
		{
			// neighbors in X axis
			if(std::abs(vecDx[disparityIdx] - vecDx[compIdx]) < windowSize)
			{
				numNeighborX++;
			}

			// neighbors in Y axis
			if(std::abs(vecDy[disparityIdx] - vecDy[compIdx]) < windowSize)
			{
				numNeighborY++;
			}
		}

		// disparity in X axis
		if(maxNeighborX < numNeighborX)
		{
			estimatedDisparity.x = vecDx[disparityIdx];
			maxNeighborX = numNeighborX;
		}

		// disparity in Y axis
		if(maxNeighborY < numNeighborY)
		{
			estimatedDisparity.y = vecDy[disparityIdx];
			maxNeighborY = numNeighborY;
		}
	}

	// find inliers
	for(size_t vectorIdx = 0; vectorIdx < numMovingFeatures; vectorIdx++)
	{
		if((vecMovingVector[vectorIdx] - estimatedDisparity).norm_L2() < windowSize)
		{
			inlierFeatureIndex.push_back(vecMovingFeatuerIdx[vectorIdx]);
		}
	}

	// estimate box
	PSN_Rect estimatedBox = preBox;
	estimatedBox.x += estimatedDisparity.x;
	estimatedBox.y += estimatedDisparity.y;

	return estimatedBox;
}


///************************************************************************
// Method Name: NSSD
// Description: 
//	- Normalized SSD between two patches
// Input Arguments:
//	- matPatch1:
//	- matPatch2:
// Return Values:
//	- result SSD
//************************************************************************/
//double CPSNWhere_Tracker2D::NSSD(cv::Mat *matPatch1, cv::Mat *matPatch2)
//{
//	cv::Mat *compPatch = matPatch2;
//	if(matPatch1->size != matPatch2->size)
//	{		
//		compPatch = new cv::Mat(matPatch1->rows, matPatch1->cols, matPatch1->type());
//		cv::resize(*matPatch2, *compPatch, cv::Size(matPatch1->rows, matPatch1->cols));
//	}
//
//	double ssd = 0.0;
//	//double invNumPixels = 1.0 / (matPatch1->channels() * matPatch1->rows * matPatch1->cols);
//	double normConstant = 0.0;
//	for(int channelIdx = 0; channelIdx < matPatch1->channels(); channelIdx++)
//	{
//		for(int rowIdx = 0; rowIdx < matPatch1->rows; rowIdx++)
//		{
//			for(int colIdx = 0; colIdx < matPatch1->cols; colIdx++)
//			{
//				if(0 == this->m_matMaskForPatch.at<float>(rowIdx, colIdx))
//				{
//					continue;
//				}
//				//ssd += invNumPixels * abs((double)matPatch1->at<unsigned char>(rowIdx, colIdx) - (double)matPatch2->at<unsigned char>(rowIdx, colIdx));
//				ssd += this->m_matMaskForPatch.at<float>(rowIdx, colIdx) * abs((double)matPatch1->at<unsigned char>(rowIdx, colIdx) - (double)matPatch2->at<unsigned char>(rowIdx, colIdx));
//				normConstant += this->m_matMaskForPatch.at<float>(rowIdx, colIdx);
//			}
//		}
//	}
//	ssd /= normConstant;
//
//	if(matPatch1->size != matPatch2->size)
//	{
//		delete compPatch;
//	}
//	return ssd;
//}


/************************************************************************
 Method Name: BoxDistance
 Description: 
	- Calculate the distance between two boxes
 Input Arguments:
	- box1: the first box
	- box2: the second box
 Return Values:
	- distance between two boxes
************************************************************************/
double CPSNWhere_Tracker2D::BoxMatchingCost(PSN_Rect &box1, PSN_Rect &box2)
{
	double nominator = (box1.center() - box2.center()).norm_L2();
	double denominator = (box1.w + box2.w)/2.0;
	double boxDistance = (nominator * nominator) / (denominator * denominator);
	double probability = std::exp(-boxDistance);

	double cost = -std::numeric_limits<double>::infinity();
	if(1.0 > probability)
	{
		cost = boxDistance + std::log(1.0 - probability); 
	}
	return cost;
}


/************************************************************************
 Method Name: GetTrackingConfidence
 Description: 
	- Calculate the confidence of tracking by the number of features lay in the box
 Input Arguments:
	- box: target position
	- vecTrackedFeatures: tracked features
 Return Values:
	- tracking confidence
************************************************************************/
double CPSNWhere_Tracker2D::GetTrackingConfidence(PSN_Rect &box, std::vector<cv::Point2f> &vecTrackedFeatures)
{
	double numFeaturesInBox = 0.0;
	for(std::vector<cv::Point2f>::iterator featureIter = vecTrackedFeatures.begin();
		featureIter != vecTrackedFeatures.end();
		featureIter++)
	{
		if(box.contain(*featureIter))
		{
			numFeaturesInBox++;
		}
	}

	return numFeaturesInBox / (double)vecTrackedFeatures.size();
}


/************************************************************************
 Method Name: MotionEstimation
 Description: 
	- Estimate the next postion with motion information
 Input Arguments:
	- tracker: tracking model
 Return Values:
	- estimated position
************************************************************************/
PSN_Point2D CPSNWhere_Tracker2D::MotionEstimation(stTracker2D &tracker)
{
	PSN_Point2D estimatedPosition(0.0, 0.0);



	return estimatedPosition;
}


/************************************************************************
 Method Name: Track2D_BackwardFeatureTracking
 Description: 
	- 
 Input Arguments:
	- 
	- 
 Return Values:
	- number of vaild detections
************************************************************************/
//#define OPTICAL_FLOW_AT_ONCE
size_t CPSNWhere_Tracker2D::Track2D_BackwardFeatureTracking(std::vector<stDetection> &curDetectionResult)
{
#ifdef OPTICAL_FLOW_AT_ONCE
	std::vector<stDetectedObject> vecDetection2D;
	if(0 == curDetectionResult.size())
	{
		this->m_vecDetection2D = vecDetection2D;
		return 0;
	}

	size_t numDetection = curDetectionResult.size();
	size_t detectionID = 0;	
	this->m_vecDetection2D.clear();
	this->m_vecDetection2D.reserve(numDetection);		
	std::vector<cv::KeyPoint> newKeypoints;
	newKeypoints.reserve(PSN_2D_FEATURE_MAX_NUM_TRACK);	
	double curHeight = 0.0;
	PSN_Point2D bottomCenter(0.0, 0.0), topCenter(0.0, 0.0);
	PSN_Point3D curLocation;

	cv::Mat *ptmatCurrGrayFrame = NULL;
	cv::Mat *ptmatPrevGrayFrame = NULL;	

	double maxDetectionWidth = 0.0;
	double maxDetectionHeight = 0.0;

	/////////////////////////////////////////////////////////////////////////////
	// DETECTION VALIDATION AND FEATURE EXTRACTION
	/////////////////////////////////////////////////////////////////////////////
	std::vector<cv::Point2f> currFeatures;
	std::vector<cv::Point2f> prevFeatures;
	std::vector<uchar> featureStatus;
	std::vector<float> featureErrors;
	std::vector<int> featuresDetectionID;
	for(size_t detectionIdx = 0; detectionIdx < numDetection; detectionIdx++)
	{
		//---------------------------------------------------
		// DETECTION VALIDATION
		//---------------------------------------------------
		// height condition
		bottomCenter = curDetectionResult[detectionIdx].box.bottomCenter();
		topCenter = bottomCenter;
		topCenter.y -= curDetectionResult[detectionIdx].box.h;
		curHeight = this->EstimateDetectionHeight(bottomCenter, topCenter, &curLocation);
		if(PSN_2D_MAX_HEIGHT < curHeight || PSN_2D_MIN_HEIGHT > curHeight)
		{
			continue;
		}

		// generate detection information
		stDetectedObject curDetection;
		curDetection.id = (unsigned int)detectionID++;
		curDetection.detection = curDetectionResult[detectionIdx];
		curDetection.bMatchedWithTracker = false;
		curDetection.vecvecTrackedFeatures.reserve(PSN_2D_BACKTRACKING_INTERVAL);
		curDetection.boxes.reserve(PSN_2D_BACKTRACKING_INTERVAL);
		curDetection.boxes.push_back(curDetection.detection.box);
		curDetection.location = curLocation;
		curDetection.height = curHeight;

		size_t costPos = detectionIdx * this->m_queueActiveTracker2D.size();

		//---------------------------------------------------
		// FEATURE EXTRACTION
		//---------------------------------------------------
		// input and mask image for current detection		
		//cv::Rect rectROI = curDetection.boxes[0].cropWithSize(this->m_nInputWidth, this->m_nInputHeight).scale(PSN_2D_OPTICALFLOW_SCALE).cv();	// full-body
		cv::Rect rectROI = curDetection.boxes[0].scale(PSN_2D_OPTICALFLOW_SCALE).cropWithSize((*this->m_vecPtGrayFrameBuffer.rbegin())->cols, (*this->m_vecPtGrayFrameBuffer.rbegin())->rows).cv();	// full-body
		this->m_matMaskForFeature(rectROI) = cv::Scalar(255); // masking with the detection box
		
		// extrack point position
		ptmatCurrGrayFrame = *(this->m_vecPtGrayFrameBuffer.rbegin());
		this->m_detector->detect(*ptmatCurrGrayFrame, newKeypoints, this->m_matMaskForFeature);
		this->m_matMaskForFeature(rectROI) = cv::Scalar(0); // restore the mask image
		
		if(PSN_2D_FEATURE_MIN_NUM_TRACK > newKeypoints.size())
		{
			continue;
		}
	
		// extrack point position
		std::vector<cv::Point2f> currFeaturesOfCurDetection;
		std::random_shuffle(newKeypoints.begin(), newKeypoints.end());
		for(size_t pointIdx = 0; pointIdx < std::min(newKeypoints.size(), (size_t)PSN_2D_FEATURE_MAX_NUM_TRACK); pointIdx++)
		{
			// saturate the number of feature points for speed-up
			currFeatures.push_back(newKeypoints[pointIdx].pt);
			currFeaturesOfCurDetection.push_back(newKeypoints[pointIdx].pt);
			featuresDetectionID.push_back((int)curDetection.id);
		}
		newKeypoints.clear();

		// keep instance
		curDetection.vecvecTrackedFeatures.push_back(currFeaturesOfCurDetection);
		vecDetection2D.push_back(curDetection);

		if(maxDetectionWidth < rectROI.width)
		{ 
			maxDetectionWidth = rectROI.width; 
		}

		if(maxDetectionHeight < rectROI.height)
		{ 
			maxDetectionHeight = rectROI.height; 
		}
	}

	if(0 == vecDetection2D.size())
	{
		this->m_vecDetection2D = vecDetection2D;
		return 0;
	}

	/////////////////////////////////////////////////////////////////////////////
	// BACKWARD TRACKING
	/////////////////////////////////////////////////////////////////////////////
	std::vector<bool> vecDetectionTracked(vecDetection2D.size(), true);
	for(std::vector<cv::Mat*>::reverse_iterator frameIter = this->m_vecPtGrayFrameBuffer.rbegin() + 1;
		frameIter != this->m_vecPtGrayFrameBuffer.rend();
		frameIter++)
	{
		ptmatPrevGrayFrame = *frameIter;
		if(NULL == ptmatPrevGrayFrame || 0 == currFeatures.size())
		{ 
			break; 
		}

		// backward feature tracking
		featureStatus.clear();
		featureErrors.clear();
		prevFeatures.clear();
		cv::calcOpticalFlowPyrLK(*ptmatCurrGrayFrame, 
								*ptmatPrevGrayFrame,
								currFeatures, 
								prevFeatures, 
								featureStatus, 
								featureErrors,
								cv::Size((int)(maxDetectionWidth * PSN_2D_FEATURE_WIN_SIZE_RATIO), (int)(maxDetectionHeight * PSN_2D_FEATURE_WIN_SIZE_RATIO)));

		// position estimation
		std::vector<cv::Point2f> newCurrFeatures;
		std::vector<int> newFeaturesDetectionID;
		int numTrackedFeatures = 0;
		int featureStartPos = 0;
		int curDetectionID = 0;
		for(size_t detectionIdx = 0; detectionIdx < vecDetection2D.size(); detectionIdx++)
		{
			if(!vecDetectionTracked[detectionIdx])
			{ 
				continue;	
			}

			std::vector<size_t> vecInlierIndex;
			stDetectedObject *curDetection = &vecDetection2D[detectionIdx];

			std::vector<cv::Point2f> currFeaturesOfCurDetection;
			std::vector<cv::Point2f> prevFeaturesOfCurDetection;

			if(featureStartPos >= (int)featuresDetectionID.size())
			{ 
				break; 
			}

			curDetectionID = featuresDetectionID[featureStartPos];
			while(featureStartPos < (int)featuresDetectionID.size() && featuresDetectionID[featureStartPos] == curDetectionID)
			{
				currFeaturesOfCurDetection.push_back(currFeatures[featureStartPos]);
				prevFeaturesOfCurDetection.push_back(prevFeatures[featureStartPos]);
				featureStartPos++;
			}
			
			PSN_Rect newRect = LocalSearchKLT(curDetection->detection.box.scale(PSN_2D_OPTICALFLOW_SCALE), currFeaturesOfCurDetection, prevFeaturesOfCurDetection, vecInlierIndex);
			if(PSN_2D_FEATURE_MIN_NUM_TRACK > vecInlierIndex.size())
			{
				vecDetectionTracked[detectionIdx] = false;
				continue;
			}
			curDetection->boxes.push_back(newRect.scale(PSN_2D_OPTICALFLOW_SCALE_RECOVER));

			// save inlier features
			if(1 == curDetection->vecvecTrackedFeatures.size())
			{
				curDetection->vecvecTrackedFeatures[0].clear();
				for(size_t indexIdx = 0; indexIdx < vecInlierIndex.size(); indexIdx++)
				{
					curDetection->vecvecTrackedFeatures[0].push_back(currFeaturesOfCurDetection[vecInlierIndex[indexIdx]]);
				}
			}

			std::vector<cv::Point2f> prevInlierFeatures;
			for(size_t indexIdx = 0; indexIdx < vecInlierIndex.size(); indexIdx++)
			{
				prevInlierFeatures.push_back(prevFeaturesOfCurDetection[vecInlierIndex[indexIdx]]);

				newCurrFeatures.push_back(prevFeaturesOfCurDetection[vecInlierIndex[indexIdx]]);
				newFeaturesDetectionID.push_back(curDetectionID);
			}
			curDetection->vecvecTrackedFeatures.push_back(prevInlierFeatures);

			numTrackedFeatures += (int)prevInlierFeatures.size();
		}

		if(0 == numTrackedFeatures)
		{
			break;
		}

		// buffer control
		ptmatCurrGrayFrame = ptmatPrevGrayFrame;
		currFeatures = newCurrFeatures;
		featuresDetectionID = newFeaturesDetectionID;
	}
	this->m_vecDetection2D = vecDetection2D;
	return this->m_vecDetection2D.size();

#else

	size_t numDetection = curDetectionResult.size();
	size_t detectionID = 0;	
	this->m_vecDetection2D.clear();
	this->m_vecDetection2D.reserve(numDetection);		
	std::vector<cv::KeyPoint> newKeypoints;
	newKeypoints.reserve(PSN_2D_FEATURE_MAX_NUM_TRACK);	
	double curHeight = 0.0;
	PSN_Point2D bottomCenter(0.0, 0.0), topCenter(0.0, 0.0);
	PSN_Point3D curLocation;

	cv::Mat *ptmatCurrGrayFrame = NULL;
	cv::Mat *ptmatPrevGrayFrame = NULL;	

	for(size_t detectionIdx = 0; detectionIdx < numDetection; detectionIdx++)
	{
		//---------------------------------------------------
		// DETECTION VALIDATION
		//---------------------------------------------------
		// height condition
		bottomCenter = curDetectionResult[detectionIdx].box.bottomCenter();
		topCenter = bottomCenter;
		topCenter.y -= curDetectionResult[detectionIdx].box.h;
		curHeight = this->EstimateDetectionHeight(bottomCenter, topCenter, &curLocation);
		if (PSN_2D_MAX_HEIGHT < curHeight || PSN_2D_MIN_HEIGHT > curHeight) { continue; }

		// generate detection information
		stDetectedObject curDetection;
		curDetection.id = (unsigned int)detectionID++;
		curDetection.detection = curDetectionResult[detectionIdx];
		curDetection.bMatchedWithTracker = false;
		curDetection.bOverlapWithOtherDetection = false;
		curDetection.vecvecTrackedFeatures.reserve(PSN_2D_BACKTRACKING_INTERVAL);
		curDetection.boxes.reserve(PSN_2D_BACKTRACKING_INTERVAL);
		curDetection.boxes.push_back(curDetection.detection.box);
		curDetection.location = curLocation;
		curDetection.height = curHeight;

		size_t costPos = detectionIdx * this->m_queueActiveTracker2D.size();

		//---------------------------------------------------
		// FEATURE EXTRACTION
		//---------------------------------------------------
		// input and mask image for current detection		
		cv::Rect rectROI = curDetection.boxes[0].scale(PSN_2D_OPTICALFLOW_SCALE).cropWithSize((*this->m_vecPtGrayFrameBuffer.rbegin())->cols, (*this->m_vecPtGrayFrameBuffer.rbegin())->rows).cv();	// full-body
		//cv::Rect rectROI = curDetection.detection.vecPartBoxes[0].cropWithSize(this->m_nInputWidth, this->m_nInputHeight).cv(); // head
		this->m_matMaskForFeature(rectROI) = cv::Scalar(255); // masking with the detection box
		
		// extrack point position
		ptmatCurrGrayFrame = *(this->m_vecPtGrayFrameBuffer.rbegin());
		this->m_detector->detect(*ptmatCurrGrayFrame, newKeypoints, this->m_matMaskForFeature);
		this->m_matMaskForFeature(rectROI) = cv::Scalar(0); // restore the mask image
		
		if (PSN_2D_FEATURE_MIN_NUM_TRACK > newKeypoints.size()) { continue; }
	
		// extrack point position
		std::vector<cv::Point2f> currFeatures;
		std::vector<cv::Point2f> prevFeatures;
		std::vector<uchar> featureStatus;
		std::vector<float> featureErrors;

		std::random_shuffle(newKeypoints.begin(), newKeypoints.end());
		for (size_t pointIdx = 0; pointIdx < std::min(newKeypoints.size(), (size_t)PSN_2D_FEATURE_MAX_NUM_TRACK); pointIdx++)
		{
			// saturate the number of feature points for speed-up
			currFeatures.push_back(newKeypoints[pointIdx].pt);
		}
		newKeypoints.clear();
	
		//---------------------------------------------------
		// BACKWARD FEATURE TRACKING
		//---------------------------------------------------
		for (std::vector<cv::Mat*>::reverse_iterator frameIter = this->m_vecPtGrayFrameBuffer.rbegin() + 1;
			frameIter != this->m_vecPtGrayFrameBuffer.rend();
			frameIter++)
		{
			ptmatPrevGrayFrame = *frameIter;
			if (NULL == ptmatPrevGrayFrame) { break;	}

			PSN_Rect rectRescaledDetectionBox = curDetection.detection.box.scale(PSN_2D_OPTICALFLOW_SCALE);

			// backward feature tracking
			featureStatus.clear();
			featureErrors.clear();
			prevFeatures.clear();
			cv::calcOpticalFlowPyrLK(*ptmatCurrGrayFrame, 
									*ptmatPrevGrayFrame,
									currFeatures, 
									prevFeatures, 
									featureStatus, 
									featureErrors,
									cv::Size((int)(rectRescaledDetectionBox.w * PSN_2D_FEATURE_WIN_SIZE_RATIO), (int)(rectRescaledDetectionBox.w * PSN_2D_FEATURE_WIN_SIZE_RATIO)));


			// position estimation
			std::vector<size_t> vecInlierIndex;
			PSN_Rect newRect = LocalSearchKLT(rectRescaledDetectionBox, currFeatures, prevFeatures, vecInlierIndex);
			if (PSN_2D_FEATURE_MIN_NUM_TRACK > vecInlierIndex.size()) {	break; }
			curDetection.boxes.push_back(newRect.scale(PSN_2D_OPTICALFLOW_SCALE_RECOVER));

			// save inlier features
			if (0 == curDetection.vecvecTrackedFeatures.size())
			{
				std::vector<cv::Point2f> currInlierFeatures;
				for (size_t indexIdx = 0; indexIdx < vecInlierIndex.size(); indexIdx++)
				{
					currInlierFeatures.push_back(currFeatures[vecInlierIndex[indexIdx]]);
				}
				curDetection.vecvecTrackedFeatures.push_back(currInlierFeatures);
			}
			currFeatures.clear();
			currFeatures.reserve(vecInlierIndex.size());
			for (size_t indexIdx = 0; indexIdx < vecInlierIndex.size(); indexIdx++)
			{
				currFeatures.push_back(prevFeatures[vecInlierIndex[indexIdx]]);
			}
			curDetection.vecvecTrackedFeatures.push_back(currFeatures);

			// buffer control
			ptmatCurrGrayFrame = ptmatPrevGrayFrame;
		}

		// for the first frame
		if (0 == curDetection.vecvecTrackedFeatures.size())
		{
			curDetection.vecvecTrackedFeatures.push_back(currFeatures);
		}

		// keep instance
		this->m_vecDetection2D.push_back(curDetection);
	}

	// check overlap
	for (int detect1Idx = 0; detect1Idx < this->m_vecDetection2D.size(); detect1Idx++)
	{
		if (this->m_vecDetection2D[detect1Idx].bOverlapWithOtherDetection) { continue; }
		for (int detect2Idx = detect1Idx + 1; detect2Idx < this->m_vecDetection2D.size(); detect2Idx++)
		{
			if (this->m_vecDetection2D[detect1Idx].detection.box.overlap(this->m_vecDetection2D[detect2Idx].detection.box))
			{
				this->m_vecDetection2D[detect1Idx].bOverlapWithOtherDetection = true;
				break;
			}
		}
	}


	return this->m_vecDetection2D.size();
#endif
}


/************************************************************************
 Method Name: Track2D_ForwardTrackingAndGetMatchingScore
 Description: 
	- 
 Input Arguments:
	- 
	- 
 Return Values:
	- none
************************************************************************/
std::vector<float> CPSNWhere_Tracker2D::Track2D_ForwardTrackingAndGetMatchingScore(void)
{
	cv::Mat *ptmatCurrGrayFrame = *(this->m_vecPtGrayFrameBuffer.rbegin());
	cv::Mat *ptmatPrevGrayFrame = *(this->m_vecPtGrayFrameBuffer.rbegin() + 1);
	size_t numDetection = this->m_vecDetection2D.size();
	std::vector<unsigned char> pointStatus;
	std::vector<float> pointError;
	std::vector<float> matchingCostArray(numDetection * this->m_queueActiveTracker2D.size(), std::numeric_limits<float>::infinity());
	for(size_t trackIdx = 0; trackIdx < this->m_queueActiveTracker2D.size(); trackIdx++)
	{
		stTracker2D *curTracker = this->m_queueActiveTracker2D[trackIdx];
		
		//---------------------------------------------------
		// FORWARD FEATURE TRACKING
		//---------------------------------------------------
		pointStatus.clear();
		pointError.clear();
		curTracker->trackedPoints.clear();
		PSN_Rect curRect = curTracker->boxes.back().scale(PSN_2D_OPTICALFLOW_SCALE);
		cv::calcOpticalFlowPyrLK(*ptmatPrevGrayFrame, 
								*ptmatCurrGrayFrame,
								curTracker->featurePoints, 
								curTracker->trackedPoints, 
								pointStatus, 
								pointError,
								cv::Size((int)(curRect.w * PSN_2D_FEATURE_WIN_SIZE_RATIO), (int)(curRect.h * PSN_2D_FEATURE_WIN_SIZE_RATIO)));

		std::vector<cv::Point2f> vecPrevFeatures, vecCurrFeatures;
		vecPrevFeatures.reserve(pointStatus.size());
		vecCurrFeatures.reserve(pointStatus.size());
		for(size_t featureIdx = 0; featureIdx < pointStatus.size(); featureIdx++)
		{
			if(!pointStatus[featureIdx])
			{
				continue;
			}
			vecPrevFeatures.push_back(curTracker->featurePoints[featureIdx]);
			vecCurrFeatures.push_back(curTracker->trackedPoints[featureIdx]);
		}

		curTracker->featurePoints = vecCurrFeatures;
		if(PSN_2D_FEATURE_MIN_NUM_TRACK > curTracker->featurePoints.size())
		{
			continue;
		}

		std::vector<size_t> vecInlierIndex;
		PSN_Rect newBox = LocalSearchKLT(curTracker->boxes.back().scale(PSN_2D_OPTICALFLOW_SCALE), vecPrevFeatures, vecCurrFeatures, vecInlierIndex);
		newBox = newBox.scale(PSN_2D_OPTICALFLOW_SCALE_RECOVER);
		curTracker->boxes.push_back(newBox);

#ifdef PSN_2D_DEBUG_DISPLAY_
		
		cv::rectangle(this->m_matDebugDisplay, newBox.cv(), CV_RGB(255, 255, 255));
		size_t inlierIdx = 0;
		size_t trackedFeatureIdx = 0;
		for(size_t pointIdx = 0; pointIdx < pointStatus.size(); pointIdx++)
		{
			if(!pointStatus[pointIdx])
			{
				continue;
			}
			cv::Point2f curPoint = (float)PSN_2D_DEBUG_DISPLAY_SCALE * PSN_2D_OPTICALFLOW_SCALE_RECOVER * vecCurrFeatures[trackedFeatureIdx];
			cv::Point2f prevPoint = (float)PSN_2D_DEBUG_DISPLAY_SCALE * PSN_2D_OPTICALFLOW_SCALE_RECOVER * vecPrevFeatures[trackedFeatureIdx];
			trackedFeatureIdx++;

			if(inlierIdx < vecInlierIndex.size() && pointIdx == vecInlierIndex[inlierIdx])
			{
				cv::line(this->m_matDebugDisplay, curPoint, prevPoint, CV_RGB(0, 250, 0));
				cv::circle(this->m_matDebugDisplay, prevPoint, 1, CV_RGB(0, 250, 0), CV_FILLED);
				inlierIdx++;
			}
			else
			{
				cv::line(this->m_matDebugDisplay, curPoint, prevPoint, CV_RGB(255, 255, 255));
				cv::circle(this->m_matDebugDisplay, prevPoint, 1, CV_RGB(255, 255, 255), CV_FILLED);
			}
			cv::circle(this->m_matDebugDisplay, curPoint, 1, CV_RGB(250, 0, 250), CV_FILLED);			
		}
#endif

		//---------------------------------------------------
		// COMPARE WITH DETECTION'S BACKTRACKING RESULT
		//---------------------------------------------------
		size_t costPos = trackIdx;
		for (std::vector<stDetectedObject>::iterator detectionIter = this->m_vecDetection2D.begin();
			detectionIter != this->m_vecDetection2D.end();
			detectionIter++, costPos += this->m_queueActiveTracker2D.size())
		{
			// validate with distance in 3D space
			if (((*detectionIter).location - curTracker->lastPosition).norm_L2() > PSN_2D_MAX_DETECTION_DISTANCE) { continue; }

			// validate with height difference
			if (std::abs((*detectionIter).height - curTracker->height) > PSN_2D_MAX_HEIGHT_DIFFERENCE) { continue; }

			// validate with backward tracking result
			if (!newBox.overlap((*detectionIter).detection.box)) { continue; }

			double boxCost = 0.0;			
			size_t lengthForCompare = std::min((size_t)PSN_2D_BACKTRACKING_INTERVAL, std::min(curTracker->boxes.size(), (*detectionIter).boxes.size()));
			size_t trackerBoxIdx = (size_t)curTracker->duration; // duration = # of box - 1			
			for (size_t boxIdx = 0; boxIdx < lengthForCompare; boxIdx++, trackerBoxIdx--)
			{
				if (!curTracker->boxes[trackerBoxIdx].overlap((*detectionIter).boxes[boxIdx])
				|| PSN_2D_BOX_MAX_DISTANCE < curTracker->boxes[trackerBoxIdx].distance((*detectionIter).boxes[boxIdx]))
				{
					boxCost = std::numeric_limits<double>::infinity();
					break;
				}
				//boxCost += BoxMatchingCost(curTracker->boxes[trackerBoxIdx], (*detectionIter).boxes[boxIdx]);
			}
			if (std::numeric_limits<double>::infinity() == boxCost)	{ continue;	}

			trackerBoxIdx = curTracker->boxes.size() - lengthForCompare;
			double forwardConfidence = GetTrackingConfidence((*detectionIter).detection.box.scale(PSN_2D_OPTICALFLOW_SCALE), curTracker->featurePoints);
			double backwardConfidence = GetTrackingConfidence(curTracker->boxes[trackerBoxIdx].scale(PSN_2D_OPTICALFLOW_SCALE), (*detectionIter).vecvecTrackedFeatures.back());
			
			if(PSN_2D_MIN_CONFIDENCE > forwardConfidence || PSN_2D_MIN_CONFIDENCE > backwardConfidence) { continue;	}
			//boxCost = - std::log(forwardConfidence) - std::log(backwardConfidence);
			//printf("ID: %d, length: %d, back: %lf, forward: %lf\n", curTracker->id, lengthForCompare, backwardConfidence, forwardConfidence);

			//double appearanceCost = 0.0;
			//double motionCost = 0.0;
			//
			//
			//size_t numPointContain = 0;
			//for(size_t indexIdx = 0; indexIdx < vecInlierIndex.size(); indexIdx++)
			//{
			//	if((*detectionIter).detection.box.contain(vecCurrFeatures[vecInlierIndex[indexIdx]]))
			//	{
			//		numPointContain++;
			//	}
			//}
			//if(!numPointContain)
			//{				
			//	continue;
			//}

			//// calculate the confidence of forward tracking
			//float trackedPointRatio = (float)numPointContain / (float)vecInlierIndex.size();
			//if(PSN_2D_COST_THRESHOLD_FEATURE > trackedPointRatio)
			//{
			//	continue;
			//}

			matchingCostArray[costPos] = (float)boxCost;
		}
	}

	return matchingCostArray;
}


/************************************************************************
 Method Name: Track2D_MatchingAndUpdating
 Description: 
	- 
 Input Arguments:
	- 
	- 
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Tracker2D::Track2D_MatchingAndUpdating(std::vector<float> &matchingCostArray)
{
	//cv::Mat *ptmatCurrGrayFrame = *(this->m_vecPtGrayFrameBuffer.rbegin());
	//cv::Mat matCurrFrame(*ptmatCurrGrayFrame);

	size_t numDetection = this->m_vecDetection2D.size();
	CPSNWhere_Hungarian cHungarianMatcher;
	cHungarianMatcher.Initialize(matchingCostArray, (unsigned int)numDetection, (unsigned int)this->m_queueActiveTracker2D.size());
	stMatchInfo *curMatchInfo = cHungarianMatcher.Match();
	for (size_t matchIdx = 0; matchIdx < curMatchInfo->rows.size(); matchIdx++)
	{
		if (std::numeric_limits<float>::infinity() == curMatchInfo->matchCosts[matchIdx]) { continue; }
		stDetectedObject *curDetection = &this->m_vecDetection2D[curMatchInfo->rows[matchIdx]];
		stTracker2D *curTracker = this->m_queueActiveTracker2D[curMatchInfo->cols[matchIdx]];	

		//---------------------------------------------------
		// MATCHING VALIDATION
		//---------------------------------------------------
		//if (PSN_2D_MIN_CONFIDENCE > curTracker->confidence) { continue; }
		//if (curDetection->bOverlapWithOtherDetection) { continue; }
		double curConfidence = 1.0;

		//---------------------------------------------------
		// TRACKER UPDATE
		//---------------------------------------------------
		curDetection->bMatchedWithTracker = true;		
		curTracker->timeEnd = this->m_nCurrentFrameIdx;
		curTracker->timeLastUpdate = this->m_nCurrentFrameIdx;
		curTracker->duration = curTracker->timeEnd - curTracker->timeStart + 1;
		curTracker->numStatic = 0;
		curTracker->boxes.back() = curDetection->detection.box;
		curTracker->featurePoints = curDetection->vecvecTrackedFeatures.front();
		curTracker->trackedPoints.clear();
		curTracker->confidence = curConfidence;
		curTracker->lastPosition = curDetection->location;
		curTracker->height = curDetection->height;
	}
	cHungarianMatcher.Finalize();	


	/////////////////////////////////////////////////////////////////////////////
	// TRACKER GENERATION
	/////////////////////////////////////////////////////////////////////////////
	for (std::vector<stDetectedObject>::iterator detectionIter = this->m_vecDetection2D.begin();
		detectionIter != this->m_vecDetection2D.end();
		detectionIter++)
	{
		if ((*detectionIter).bMatchedWithTracker)
		{
//#ifdef PSN_2D_DEBUG_DISPLAY_
//			PSN_Rect curBox = (*detectionIter).detection.box;
//			curBox.x *= PSN_2D_DEBUG_DISPLAY_SCALE;
//			curBox.y *= PSN_2D_DEBUG_DISPLAY_SCALE;
//			curBox.w *= PSN_2D_DEBUG_DISPLAY_SCALE;
//			curBox.h *= PSN_2D_DEBUG_DISPLAY_SCALE;
//			cv::rectangle(testMat, curBox.cv(), cv::Scalar(255, 255, 255));
//#endif
			continue;
		}
		
		stTracker2D newTracker;
		newTracker.id = this->m_nNewTrackerID++;
		newTracker.timeStart = this->m_nCurrentFrameIdx;
		newTracker.timeEnd = this->m_nCurrentFrameIdx;
		newTracker.timeLastUpdate = this->m_nCurrentFrameIdx;
		newTracker.duration = 1;
		newTracker.numStatic = 0;
		newTracker.boxes.push_back((*detectionIter).detection.box);
		newTracker.featurePoints = (*detectionIter).vecvecTrackedFeatures.front();
		newTracker.trackedPoints.clear();
		//newTracker.confidence = (*detectionIter).bOverlapWithOtherDetection ? 0.0 : 1.0;
		newTracker.confidence = 1.0;
		newTracker.lastPosition = (*detectionIter).location;
		newTracker.height = (*detectionIter).height;
		
		// appearance model
		//newTracker.appModel = cv::Mat(PSN_2D_APP_PATCH_HEIGHT, PSN_2D_APP_PATCH_WIDTH, CV_8UC3);

		// insert to the data structure
		this->m_listTracker2D.push_back(newTracker);
		this->m_queueActiveTracker2D.push_back(&this->m_listTracker2D.back());
	}


	/////////////////////////////////////////////////////////////////////////////
	// TRACKER TERMINATION & RESULT PACKING
	/////////////////////////////////////////////////////////////////////////////
	this->m_stTrack2DResult.object2DInfos.clear();
	std::deque<stTracker2D*> queueNewActiveTrackers;
	for (std::deque<stTracker2D*>::iterator trackerIter = this->m_queueActiveTracker2D.begin();
		trackerIter != this->m_queueActiveTracker2D.end();
		trackerIter++)
	{
		if ((*trackerIter)->timeLastUpdate == this->m_nCurrentFrameIdx)
		{
			PSN_Rect curBox = (*trackerIter)->boxes.back();
			curBox.x *= (float)PSN_2D_DEBUG_DISPLAY_SCALE;
			curBox.y *= (float)PSN_2D_DEBUG_DISPLAY_SCALE;
			curBox.w *= (float)PSN_2D_DEBUG_DISPLAY_SCALE;
			curBox.h *= (float)PSN_2D_DEBUG_DISPLAY_SCALE;
			
#ifdef PSN_2D_DEBUG_DISPLAY_			
			//-----------------------
			// DEBUG
			//-----------------------
			//cv::rectangle(testMat, (*trackerIter)->boxes.back().cv(), cv::Scalar(255, 255, 255), 1);
			psn::DrawBoxWithID(this->m_matDebugDisplay, curBox, (*trackerIter)->id, this->m_vecColors);
#endif
			queueNewActiveTrackers.push_back(*trackerIter);

			// result packing
			stObject2DInfo objectInfo;
			objectInfo.id = (*trackerIter)->id;
			objectInfo.box = curBox;
			objectInfo.score = 0;
			this->m_stTrack2DResult.object2DInfos.push_back(objectInfo);

			continue;
		}
		
		// termination
		(*trackerIter)->boxes.clear();
		(*trackerIter)->featurePoints.clear();
		(*trackerIter)->trackedPoints.clear();
		//(*trackerIter)->appModel.release();
	}
	this->m_queueActiveTracker2D = queueNewActiveTrackers;

	// matching cost
	int cost_pos = 0;
	if (!this->m_stTrack2DResult.matMatchingCost.empty()) 
	{
		this->m_stTrack2DResult.matMatchingCost.release();
	}
	
	this->m_stTrack2DResult.matMatchingCost = cv::Mat((int)m_vecDetection2D.size(), (int)this->m_stTrack2DResult.vecTrackerRects.size(), CV_32F);
	for (int detectionIdx = 0; detectionIdx < m_vecDetection2D.size(); detectionIdx++) 
	{
		for (int trackIdx = 0; trackIdx < this->m_stTrack2DResult.vecTrackerRects.size(); trackIdx++)	
		{
			this->m_stTrack2DResult.matMatchingCost.at<float>(detectionIdx, trackIdx) = matchingCostArray[cost_pos];
			cost_pos++;
		}
	}
}


/************************************************************************
 Method Name: EstimateDetectionHeight
 Description: 
	- Estimate a height in 3D space
 Input Arguments:
	- bottomCenter:
	- topCenter:
 Return Values:
	- height in mm unit
************************************************************************/
double CPSNWhere_Tracker2D::EstimateDetectionHeight(PSN_Point2D bottomCenter, PSN_Point2D topCenter, PSN_Point3D *location3D)
{
	PSN_Point3D P11, P12, P21, P22;

	// top point
	P11.z = 0;
	P12.z = 2000;
	this->m_stCalibrationInfos.cCamModel.imageToWorld(topCenter.x, topCenter.y, P11.z, P11.x, P11.y);
	this->m_stCalibrationInfos.cCamModel.imageToWorld(topCenter.x, topCenter.y, P12.z, P12.x, P12.y);

	// bottom point
	P21.z = 0;	
	this->m_stCalibrationInfos.cCamModel.imageToWorld(bottomCenter.x, bottomCenter.y, P21.z, P21.x, P21.y);
	P22 = P21;
	P22.z = 2000;

	if(NULL != location3D)
	{
		*location3D = P21;
	}

	PSN_Point3D topPoint(0.0, 0.0, 0.0);
	CPSNWhere_Manager::Triangulation(PSN_Line(P11, P12), PSN_Line(P21, P22), topPoint);

	return (topPoint - P21).norm_L2();
}


/************************************************************************
 Method Name: FilePrintTracklet
 Description: 
	- Save current tracklet information to the file
 Input Arguments:
	- none	
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Tracker2D::FilePrintTracklet(void)
{
	char strFilePathAndName[128];
	FILE *fp;
	try
	{		
		sprintf_s(strFilePathAndName, "%stracklets/track2D_result_cam%d_frame%04d.txt", RESULT_SAVE_PATH, (int)this->m_nCamID, (int)this->m_nCurrentFrameIdx);
		fopen_s(&fp, strFilePathAndName, "w");

		fprintf_s(fp, "camIdx:%d\nframeIdx:%d\nnumModel:%d\n", (int)this->m_nCamID, (int)this->m_nCurrentFrameIdx, (int)this->m_queueActiveTracker2D.size());

		for(std::deque<stTracker2D*>::iterator trackerIter = this->m_queueActiveTracker2D.begin();
			trackerIter != this->m_queueActiveTracker2D.end();
			trackerIter++)
		{
			PSN_Rect curBox = (*trackerIter)->boxes.back();
			fprintf_s(fp, "{id:%d,box:{%d,%d,%d,%d}}\n", (int)(*trackerIter)->id, (int)curBox.x, (int)curBox.y, (int)curBox.w, (int)curBox.h);
		}
		
		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](FilePrintTracklet) cannot open file! error code %d\n", dwError);
		return;
	}
}

/************************************************************************
 Method Name: SaveSnapshot
 Description: 
	- 
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPSNWhere_Tracker2D::SaveSnapshot(const char *strFilepath)
{
	FILE *fp;
	char strFilename[128] = "";
	try
	{	
		sprintf_s(strFilename, "%ssnapshot_2D_cam%d.txt", strFilepath, m_nCamID);
		fopen_s(&fp, strFilename, "w");
		fprintf_s(fp, "camID:%d\n", m_nCamID);
		fprintf_s(fp, "frameIndex:%d\n\n", (int)m_nCurrentFrameIdx);
		
		// tracker list
		fprintf_s(fp, "listTracker2D:%d,\n{\n", m_listTracker2D.size());
		for (std::list<stTracker2D>::iterator trackerIter = m_listTracker2D.begin();
			trackerIter != m_listTracker2D.end();
			trackerIter++)
		{
			fprintf_s(fp, "\t{\n");
			fprintf_s(fp, "\t\tid:%d\n", (int)(*trackerIter).id);
			fprintf_s(fp, "\t\ttimeStart:%d\n", (int)(*trackerIter).timeStart);
			fprintf_s(fp, "\t\ttimeEnd:%d\n", (int)(*trackerIter).timeEnd);
			fprintf_s(fp, "\t\ttimeLastUpdate:%d\n", (int)(*trackerIter).timeLastUpdate);
			fprintf_s(fp, "\t\tduration:%d\n", (int)(*trackerIter).duration);
			fprintf_s(fp, "\t\tnumStatic:%d\n", (int)(*trackerIter).numStatic);
			fprintf_s(fp, "\t\tconfidence:%e\n", (*trackerIter).confidence);

			// box
			fprintf_s(fp, "\t\tboxes:%d,{", (int)(*trackerIter).boxes.size());
			for (int boxIdx = 0; boxIdx < (*trackerIter).boxes.size(); boxIdx++)
			{
				PSN_Rect curBox = (*trackerIter).boxes[boxIdx];
				fprintf_s(fp, "(%f,%f,%f,%f)", curBox.x, curBox.y, curBox.w, curBox.h);
				if (boxIdx < (*trackerIter).boxes.size() - 1) { fprintf_s(fp, ","); }
			}
			fprintf_s(fp, "}\n");

			// featurePoints
			fprintf_s(fp, "\t\tfeaturePoints:%d,{", (int)(*trackerIter).featurePoints.size());
			for (int pointIdx = 0; pointIdx < (*trackerIter).featurePoints.size(); pointIdx++)
			{
				cv::Point2f curPoint = (*trackerIter).featurePoints[pointIdx];
				fprintf_s(fp, "(%f,%f)", curPoint.x, curPoint.y);
				if (pointIdx < (*trackerIter).featurePoints.size() - 1) { fprintf_s(fp, ","); }
			}
			fprintf_s(fp, "}\n");

			// trackedPoints
			fprintf_s(fp, "\t\ttrackedPoints:%d,{", (int)(*trackerIter).trackedPoints.size());
			for (int pointIdx = 0; pointIdx < (*trackerIter).trackedPoints.size(); pointIdx++)
			{
				cv::Point2f curPoint = (*trackerIter).trackedPoints[pointIdx];
				fprintf_s(fp, "(%f,%f,%f,%f)", curPoint.x, curPoint.y);
				if (pointIdx < (*trackerIter).trackedPoints.size() - 1) { fprintf_s(fp, ","); }
			}
			fprintf_s(fp, "}\n");

			// last point
			fprintf_s(fp, "\t\tlastPosition:(%f,%f,%f)\n", (*trackerIter).lastPosition.x, (*trackerIter).lastPosition.y, (*trackerIter).lastPosition.z);

			// height
			fprintf_s(fp, "\t\theight:%e\n\t}\n", (*trackerIter).height);
		}
		fprintf_s(fp, "}\n");

		// active tracker queue
		fprintf_s(fp, "queueActiveTracker2D:%d,{", m_queueActiveTracker2D.size());
		for (int trackerIdx = 0; trackerIdx < m_queueActiveTracker2D.size(); trackerIdx++)
		{
			fprintf_s(fp, "%d", m_queueActiveTracker2D[trackerIdx]->id);
			if (trackerIdx < m_queueActiveTracker2D.size() - 1) { fprintf_s(fp, ","); }
		}
		fprintf_s(fp, "}\n");

		fprintf_s(fp, "NewTrackerID:%d\n", (int)m_nNewTrackerID);

		fprintf_s(fp, "()()\n");
		fprintf_s(fp, "('')\n");

		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](PrintTracks) cannot open file! error code %d\n", dwError);
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
	- none
************************************************************************/
bool CPSNWhere_Tracker2D::LoadSnapshot(const char *strFilepath)
{
	FILE *fp;
	char strFilename[128] = "";
	int readingInt = 0;
	float readingFloat = 0.0;
	try
	{	
		sprintf_s(strFilename, "%ssnapshot_2D_cam%d.txt", strFilepath, m_nCamID);
		fopen_s(&fp, strFilename, "r");
		if (NULL == fp) { return false; }

		fscanf_s(fp, "camID:%d\n", &readingInt);
		fscanf_s(fp, "frameIndex:%d\n\n", &readingInt);
		m_nCurrentFrameIdx = (unsigned int)readingInt;
		
		// tracker list
		m_listTracker2D.clear();
		int numTracker = 0;
		fscanf_s(fp, "listTracker2D:%d,\n{\n", &numTracker);
		for (int trackerIdx = 0; trackerIdx < numTracker; trackerIdx++)
		{
			stTracker2D newTracker;
			fscanf_s(fp, "\t{\n\t\tid:%d\n", &readingInt); 
			newTracker.id = (unsigned int)readingInt;
			fscanf_s(fp, "\t\ttimeStart:%d\n", &readingInt); 
			newTracker.timeStart = (unsigned int)readingInt;
			fscanf_s(fp, "\t\ttimeEnd:%d\n", &readingInt); 
			newTracker.timeEnd = (unsigned int)readingInt;
			fscanf_s(fp, "\t\ttimeLastUpdate:%d\n", &readingInt); 
			newTracker.timeLastUpdate = (unsigned int)readingInt;
			fscanf_s(fp, "\t\tduration:%d\n", &readingInt); 
			newTracker.duration = (unsigned int)readingInt;
			fscanf_s(fp, "\t\tnumStatic:%d\n", &readingInt); 
			newTracker.numStatic = (unsigned int)readingInt;
			fscanf_s(fp, "\t\tconfidence:%e\n", &readingFloat);
			newTracker.confidence = (double)readingFloat;

			// box
			int numBox = 0;
			fscanf_s(fp, "\t\tboxes:%d,{", &numBox);
			for (int boxIdx = 0; boxIdx < numBox; boxIdx++)
			{				
				float x = 0.0, y = 0.0, w = 0.0, h = 0.0;
				fscanf_s(fp, "(%f,%f,%f,%f)", &x, &y, &w, &h);
				PSN_Rect curBox((double)x, (double)y, (double)w, (double)h);
				newTracker.boxes.push_back(curBox);
				if (boxIdx < numBox - 1) { fscanf_s(fp, ","); }
			}

			// featurePoints
			int numPoint = 0;
			fscanf_s(fp, "}\n\t\tfeaturePoints:%d,{", &numPoint);
			for (int pointIdx = 0; pointIdx < numPoint; pointIdx++)
			{
				cv::Point2f curPoint;
				fscanf_s(fp, "(%f,%f)", &curPoint.x, &curPoint.y);
				newTracker.featurePoints.push_back(curPoint);
				if (pointIdx < numPoint - 1) { fscanf_s(fp, ","); }
			}

			// trackedPoints
			fscanf_s(fp, "}\n\t\ttrackedPoints:%d,{", &numPoint);
			for (int pointIdx = 0; pointIdx < numPoint; pointIdx++)
			{
				cv::Point2f curPoint;
				fscanf_s(fp, "(%f,%f)", &curPoint.x, &curPoint.y);
				newTracker.trackedPoints.push_back(curPoint);
				if (pointIdx < numPoint - 1) { fscanf_s(fp, ","); }
			}

			// last point
			float x = 0.0, y = 0.0, z = 0.0;
			fscanf_s(fp, "}\n\t\tlastPosition:(%f,%f,%f)\n", &x, &y, &z);
			newTracker.lastPosition = PSN_Point3D((double)x, (double)y, (double)z);

			// height
			fscanf_s(fp, "\t\theight:%e\n\t}\n", &readingFloat);
			newTracker.height = (double)readingFloat;

			m_listTracker2D.push_back(newTracker);
		}

		// active tracker queue
		int numActiveTracker = 0;
		fscanf_s(fp, "}\nqueueActiveTracker2D:%d,{", &numActiveTracker);
		for (int trackerIdx = 0; trackerIdx < numActiveTracker; trackerIdx++)
		{
			fscanf_s(fp, "%d", &readingInt);

			// find target tracker
			for (std::list<stTracker2D>::iterator trackerIter = m_listTracker2D.begin();
				trackerIter != m_listTracker2D.end();
				trackerIter++)
			{
				if ((*trackerIter).id != (unsigned int)readingInt) { continue; }
				m_queueActiveTracker2D.push_back(&(*trackerIter));
				break;
			}
			if (trackerIdx < numActiveTracker - 1) { fprintf_s(fp, ","); }
		}

		fscanf_s(fp, "}\nNewTrackerID:%d\n", &readingInt);
		m_nNewTrackerID = (unsigned int)readingInt;

		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](PrintTracks) cannot open file! error code %d\n", dwError);
		return false;
	}

	return true;
}

//()()
//('')HAANJU.YOO


