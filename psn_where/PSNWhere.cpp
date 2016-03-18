#include "PSNWhere.h"
#include <fstream>
#include <limits>
#include <omp.h>
#include "PSNWhere_Tracker2D.h"
#include "PSNWhere_Associator3D.h"

CPSNWhere::CPSNWhere(void)
	: m_bInit(false)
{
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		this->m_cTracker2D[camIdx] = NULL;
	}
	this->m_cAssociator3D = NULL;
	this->m_bOutputVideoInit = false;
}


CPSNWhere::~CPSNWhere(void)
{
}

/*******************************************************************
 Name: Initialize
 Input:
	- void
 Return:
	- [bool] 초기화 성공 여부
 Function:
	- 변수 초기화
	- 기하 정보 입력
*******************************************************************/
bool CPSNWhere::Initialize(std::string datasetPath, stConfiguration_Associator3D *stConfig3D)
{
	if(this->m_bInit)
	{
		return false;
	}
	sprintf_s(this->m_strDatasetPath, "%s", datasetPath.c_str());


	///////////////////////////////////////////////////////////
	// CAMERA MODEL INITIALIZATION
	///////////////////////////////////////////////////////////
	std::ifstream is;
	std::vector<stCalibrationInfo*> vecCalibInfos;
	//Etiseo::UtilXml::Init();
#pragma omp parallel for
	for(int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		this->m_stCalibrationInfos[camIdx].nCamIdx = camIdx;

		//----------------------------------------------------
		// READ CALIBRATION INFORMATION
		//----------------------------------------------------
		char strCalibrationFilePath[128];
#ifdef PSN_DEBUG_MODE_
		std::cout << "Reading calibration information of camera " << CAM_ID[camIdx] <<"..." << std::endl;
#endif

		switch(PSN_INPUT_TYPE)
		{
		case 1:
			// PETS2009
			sprintf_s(strCalibrationFilePath, "%s%sView_%03d.xml", 
				this->m_strDatasetPath, 
				CALIBRATION_PATH, 
				CAM_ID[camIdx]);
			break;
		default:
			// ETRI
			sprintf_s(strCalibrationFilePath, "%s%scalibInfo_ETRI_TESTBED_cam%d.xml", 
				this->m_strDatasetPath, 
				CALIBRATION_PATH,
				CAM_ID[camIdx]);
			break;
		}
		

		if(this->m_stCalibrationInfos[camIdx].cCamModel.fromXml(strCalibrationFilePath))
		{
#ifdef PSN_DEBUG_MODE_
			std::cout << " done ..." << std::endl;
			std::cout << " Camera position : ( " << this->m_stCalibrationInfos[camIdx].cCamModel.cposx() 
				<< " , " << this->m_stCalibrationInfos[camIdx].cCamModel.cposy() << " , " 
				<< this->m_stCalibrationInfos[camIdx].cCamModel.cposz() << " )" << std::endl;
#endif
		}
		else
		{
#ifdef PSN_DEBUG_MODE_
			std::cout << " fail!" << std::endl;
#endif
		}

		//----------------------------------------------------
		// READ PROJECTION SENSITIVITY MATRIX
		//----------------------------------------------------
#ifdef PSN_DEBUG_MODE_
		std::cout << " Read projection sensitivity : ";
#endif
		if(this->ReadProjectionSensitivity(this->m_stCalibrationInfos[camIdx].matProjectionSensitivity, camIdx))
		{
#ifdef PSN_DEBUG_MODE_
			std::cout << " success" << std::endl;
#endif
		}
		else
		{
#ifdef PSN_DEBUG_MODE_
			std::cout << " fail" << std::endl;
#endif
		}	

		//----------------------------------------------------
		// READ DISTANCE FROM BOUNDARY MATRIX
		//----------------------------------------------------	
#ifdef PSN_DEBUG_MODE_
		std::cout << " Read distance from boundary : ";
#endif
		if(this->ReadDistanceFromBoundary(this->m_stCalibrationInfos[camIdx].matDistanceFromBoundary, camIdx))
		{
#ifdef PSN_DEBUG_MODE_
			std::cout << " success" << std::endl;
#endif
		}
		else
		{
#ifdef PSN_DEBUG_MODE_
			std::cout << " fail" << std::endl;
#endif
		}

		vecCalibInfos.push_back(&this->m_stCalibrationInfos[camIdx]);
	}


	///////////////////////////////////////////////////////////
	// SUB-MODULE INITIALIZATION
	///////////////////////////////////////////////////////////
	// CPSNWhere_Tracker2D	
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (NULL == this->m_cTracker2D[camIdx]) { this->m_cTracker2D[camIdx] = new CPSNWhere_Tracker2D; }
		this->m_cTracker2D[camIdx]->Initialize(camIdx, &this->m_stCalibrationInfos[camIdx]);
	}

	// CPSWhere_Associator3D
	if(NULL == this->m_cAssociator3D)
	{ 
		this->m_cAssociator3D = new CPSNWhere_Associator3D; 
	}
	this->m_cAssociator3D->Initialize(datasetPath, vecCalibInfos, stConfig3D);


	///////////////////////////////////////////////////////////
	// ETC
	///////////////////////////////////////////////////////////
	// display related
	//cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
	this->m_vecColors = psn::GenerateColors(400);
	this->m_fProcessingTime = 0.0;

#ifdef SHOW_TOPVIEW
	char strTopViewPath[256];
	//sprintf_s(strTopViewPath, "%s/topView/topview_gray.png", datasetPath.c_str());
	sprintf_s(strTopViewPath, "%s/topView/topview_gray_zoom2.png", datasetPath.c_str());
	this->m_matTopViewBase = cv::imread(strTopViewPath, cv::IMREAD_COLOR);
	cv::namedWindow("topView", cv::WINDOW_AUTOSIZE);
#endif

	this->m_bInit = true;

	return true;
}

/*******************************************************************
 Name: Finalize
 Input:
	- void
 Return:
	- [bool] 종료 작업 성공 여부
 Function:
	- 정상 종료 여부
*******************************************************************/
void CPSNWhere::Finalize()
{
	if (!this->m_bInit) { return; }

	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (NULL != this->m_cTracker2D[camIdx])
		{ 
			this->m_cTracker2D[camIdx]->Finalize();
			delete m_cTracker2D[camIdx];
			this->m_cTracker2D[camIdx] = NULL;
		}

		this->m_stCalibrationInfos[camIdx].matDistanceFromBoundary.release();
		this->m_stCalibrationInfos[camIdx].matDistanceFromBoundary.release();
	}

	if (NULL != this->m_cAssociator3D)
	{ 
		this->m_cAssociator3D->Finalize();
		delete this->m_cAssociator3D; 
		this->m_cAssociator3D = NULL;
	}

	// display related
	//cv::destroyWindow("result");
	this->m_vecColors.clear();

#ifdef SHOW_TOPVIEW
	//cv::destroyWindow("topView");
	//this->m_matTopViewBase.release();
#endif

#ifdef DO_RECORD
	if(this->m_bOutputVideoInit)
	{
		cvReleaseVideoWriter(&this->m_vwOutputVideo);
#ifdef SHOW_TOPVIEW
		cvReleaseVideoWriter(&this->m_vwOutputVideo_topView);
#endif
	}
#endif

	this->m_bInit = false;
}

/*******************************************************************
 Name: TrackPeople
 Input:
	- [CDibArray *] pDibArray: 동시에 캡쳐한 입력 카메라 영상 배열
 Return:
	- [TrackInfoArray *]: 추적 정보
 Function:
	- 입력 영상에서 물체 탐지
	- 기 소유한 트래킹 모델 업데이트
*******************************************************************/
TrackInfoArray* CPSNWhere::TrackPeople(cv::Mat *pDibArray, int frameIdx)
{
#ifdef PSN_DEBUG_MODE_
	printf("frame %04d...", frameIdx);
#endif
	clock_t timer_start;
	clock_t timer_end;
	this->m_fProcessingTime = 0;

	std::vector<stDetection> detectionResult;
	std::vector<stTrack2DResult> result2D;
	stTrack3DResult result3D;

	timer_start = clock();
#pragma omp parallel for
	for(int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{			
		// TODO: receive detection result(밑의 예시는 file로 읽어오는 것)	
		detectionResult = psn::ReadDetectionResultWithTxt(this->m_strDatasetPath, CAM_ID[camIdx], frameIdx);
		
		// 2D tracklet
		result2D.push_back(this->m_cTracker2D[camIdx]->Run(detectionResult, &pDibArray[camIdx], frameIdx));
		//result2D.push_back(psn::Read2DTrackResultWithTxt(this->m_strDatasetPath + std::string(TRACKLET_PATH), camIdx, frameIdx));
	}
	
	// 3D association
	result3D = m_cAssociator3D->Run(result2D, pDibArray, frameIdx);

	// measuring processing time
	timer_end = clock();
	this->m_fProcessingTime += (double)(timer_end - timer_start)/CLOCKS_PER_SEC;

	// visualize tracking result
	this->Visualize(pDibArray, frameIdx, result2D, result3D, 1);

#ifdef PSN_DEBUG_MODE_
	printf("%02d targets, %f sec elapsed\n", (int)result3D.object3DInfo.size(), this->m_fProcessingTime);
#endif

	return (TrackInfoArray *)0;
}

/*******************************************************************
 Name: Visualize
 Input:
	- pDibArray: input frames
	- frameIdx: time index of current frame
	- result2D: results from 2D tracker of each camera
	- result3D: result of 3D association
	- nDispMode: display mode. 
		- 0: default
		- 1: with 2D tracklet
		- 2: with detections (association information))
 Return:
	- void
 Function:
	- Visualize the result of 3D tracking
*******************************************************************/
void CPSNWhere::Visualize(cv::Mat *pDibArray, int frameIdx, std::vector<stTrack2DResult> &result2D, stTrack3DResult &result3D, int nDispMode)
{
	//---------------------------------------------------
	// DISPLAY ON INPUT IMAGE
	//---------------------------------------------------
	std::vector<cv::Mat> tileInputArray;
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++) {	
		this->m_matResultFrames[camIdx] = pDibArray[camIdx].clone();

		// display KLT tracking result
		if (3 <= nDispMode) {
			for (int trackletIdx = 0; trackletIdx < result2D[camIdx].object2DInfos.size(); trackletIdx++) {
				stObject2DInfo *curTracklet = &result2D[camIdx].object2DInfos[trackletIdx];
				if (0 == curTracklet->featurePointsCurr.size()) { continue; }
				for (int pointIdx = 0; pointIdx < curTracklet->featurePointsPrev.size(); pointIdx++) {
					cv::line(m_matResultFrames[camIdx], curTracklet->featurePointsPrev[pointIdx], curTracklet->featurePointsCurr[pointIdx], cv::Scalar(255, 255, 255), 1);
					cv::circle(m_matResultFrames[camIdx], curTracklet->featurePointsCurr[pointIdx], 1, cv::Scalar(0, 255, 0), 1);
				}
			}
		}

		// display detections
		if (2 <= nDispMode) {
			for (int detectIdx = 0; detectIdx < (int)result2D[camIdx].vecDetectionRects.size(); detectIdx++) {
				PSN_Rect curDetectBox = result2D[camIdx].vecDetectionRects[detectIdx];				
				cv::Point curDetectBoxCenter = curDetectBox.center().cv();
				cv::rectangle(m_matResultFrames[camIdx], curDetectBox.cv(), cv::Scalar(255, 255, 255), 1);
			}
		}

		// display 2D tracklets
		if (1 <= nDispMode)	{
			for (size_t trackletIdx = 0; trackletIdx < result2D[camIdx].object2DInfos.size(); trackletIdx++) {
				stObject2DInfo *curTracklet2D = &result2D[camIdx].object2DInfos[trackletIdx];		
				psn::DrawBoxWithID(m_matResultFrames[camIdx], curTracklet2D->box, curTracklet2D->id, 1, 0, &m_vecColors);
				cv::rectangle(m_matResultFrames[camIdx], curTracklet2D->box.cv(), cv::Scalar(255, 255, 255), 1); // overlay the white box
				cv::rectangle(m_matResultFrames[camIdx], curTracklet2D->head.cv(), cv::Scalar(255, 255, 255), 1);
			}
		}
	
		// display 3D trajectory result
		for (size_t trajectoryIdx = 0; trajectoryIdx < result3D.object3DInfo.size(); trajectoryIdx++) {
			stObject3DInfo *curTrajectory = &result3D.object3DInfo[trajectoryIdx];
			if (curTrajectory->bVisibleInViews[camIdx]) {
				psn::DrawBoxWithID(m_matResultFrames[camIdx], curTrajectory->rectInViews[camIdx], curTrajectory->id, 1, 1, &this->m_vecColors);
			} else {
				psn::DrawBoxWithID(m_matResultFrames[camIdx], curTrajectory->rectInViews[camIdx], curTrajectory->id, 0, 1, &this->m_vecColors);
			}
			psn::DrawLine(m_matResultFrames[camIdx], curTrajectory->recentPoint2Ds[camIdx], curTrajectory->id, 1, &this->m_vecColors);
		}

		// image tiling
		tileInputArray.push_back(this->m_matResultFrames[camIdx]);
		this->m_matResultFrames[camIdx].release();
	}
	// tiling images
	cv::Mat displayMat = psn::MakeMatTile(&tileInputArray, 2, 2);		
	// writing frame info
	char strFrameInfo[100];
	sprintf_s(strFrameInfo, "Frame: %04d", frameIdx);
	cv::rectangle(displayMat, cv::Rect(5, 2, 145, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(displayMat, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));
	// writing processing time
	sprintf_s(strFrameInfo, "processing time: %f", this->m_fProcessingTime);
	cv::rectangle(displayMat, cv::Rect(5, 30, 175, 18), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(displayMat, strFrameInfo, cv::Point(6, 42), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
	// show image
	cv::imshow("result", displayMat);

#ifdef SHOW_TOPVIEW
	//---------------------------------------------------
	// DISPLAY ON TOPVIEW
	//---------------------------------------------------
	bool bZoomed = true;
	cv::Mat topViewResult = this->m_matTopViewBase.clone();

	// writing frame info
	sprintf_s(strFrameInfo, "Frame: %04d", frameIdx);
	cv::rectangle(topViewResult, cv::Rect(5, 2, 145, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(topViewResult, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));

	// crop zone
	PSN_Point3D cropZoneCorner[4];
	PSN_Point2D cropZoneCornerOnTopView[4];
	cropZoneCorner[0] = PSN_Point3D(CROP_ZONE_X_MIN, CROP_ZONE_Y_MIN, 0.0);
	cropZoneCorner[1] = PSN_Point3D(CROP_ZONE_X_MIN, CROP_ZONE_Y_MAX, 0.0);
	cropZoneCorner[2] = PSN_Point3D(CROP_ZONE_X_MAX, CROP_ZONE_Y_MAX, 0.0);
	cropZoneCorner[3] = PSN_Point3D(CROP_ZONE_X_MAX, CROP_ZONE_Y_MIN, 0.0);
	for (int cornerIdx = 0; cornerIdx < 4; cornerIdx++)
	{
		cropZoneCornerOnTopView[cornerIdx] = psn::GetLocationOnTopView_PETS2009(cropZoneCorner[cornerIdx], bZoomed);
	}
	cv::line(topViewResult, cropZoneCornerOnTopView[0].cv(), cropZoneCornerOnTopView[1].cv(), cv::Scalar(150, 150, 150));
	cv::line(topViewResult, cropZoneCornerOnTopView[1].cv(), cropZoneCornerOnTopView[2].cv(), cv::Scalar(150, 150, 150));
	cv::line(topViewResult, cropZoneCornerOnTopView[2].cv(), cropZoneCornerOnTopView[3].cv(), cv::Scalar(150, 150, 150));
	cv::line(topViewResult, cropZoneCornerOnTopView[3].cv(), cropZoneCornerOnTopView[0].cv(), cv::Scalar(150, 150, 150));

	// draw 3D track
	for (int trajectoryIdx = 0; trajectoryIdx < result3D.object3DInfo.size(); trajectoryIdx++)
	{
		stObject3DInfo *curTrajectory = &result3D.object3DInfo[trajectoryIdx];

		// trajectory
		std::vector<PSN_Point2D> topViewRecentPoint2Ds(curTrajectory->recentPoints.size());
		for (int pointIdx = 0; pointIdx < curTrajectory->recentPoints.size(); pointIdx++)
		{
			topViewRecentPoint2Ds[pointIdx] = psn::GetLocationOnTopView_PETS2009(curTrajectory->recentPoints[pointIdx], bZoomed);
		}
		psn::DrawLine(topViewResult, topViewRecentPoint2Ds, curTrajectory->id, this->m_vecColors, 1);

		// draw ID and triangle
		psn::DrawTriangleWithID(topViewResult, topViewRecentPoint2Ds.front(), curTrajectory->id, this->m_vecColors);

		// draw 2D detections
		for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			if (0 == curTrajectory->curDetectionPosition[camIdx].x &&
				0 == curTrajectory->curDetectionPosition[camIdx].y &&
				0 == curTrajectory->curDetectionPosition[camIdx].z)
			{
				continue;
			}
			PSN_Point2D detectionLocation = psn::GetLocationOnTopView_PETS2009(curTrajectory->curDetectionPosition[camIdx], bZoomed);
			cv::circle(topViewResult, detectionLocation.cv(), 3, psn::getColorByID(this->m_vecColors, curTrajectory->id));
			cv::line(topViewResult, topViewRecentPoint2Ds.front().cv(), detectionLocation.cv(), psn::getColorByID(this->m_vecColors, curTrajectory->id));
		}
	}
	cv::imshow("topView", topViewResult);

#endif

	cv::waitKey(5);	// for showing, no delay, no display

	//---------------------------------------------------
	// RECORD
	//---------------------------------------------------
#ifdef DO_RECORD	
	IplImage *currentFrame = new IplImage(displayMat);
#ifdef SHOW_TOPVIEW
	IplImage *currentTopFrame = new IplImage(topViewResult);
#endif
	if (!this->m_bOutputVideoInit) {
		// logging related
		time_t curTimer = time(NULL);
		struct tm timeStruct;
		localtime_s(&timeStruct, &curTimer);
		char resultFileDate[256];
		char resultOutputFileName[256];
		sprintf_s(resultFileDate, "%sresult_video_%02d%02d%02d_%02d%02d%02d", 
			RESULT_SAVE_PATH,
			timeStruct.tm_year + 1900, 
			timeStruct.tm_mon+1, 
			timeStruct.tm_mday, 
			timeStruct.tm_hour, 
			timeStruct.tm_min, 
			timeStruct.tm_sec);

		sprintf_s(resultOutputFileName, "%s.avi", resultFileDate);
		this->m_vwOutputVideo = cvCreateVideoWriter(resultOutputFileName, CV_FOURCC('M','J','P','G'), 15, cvGetSize(currentFrame), 1);
#ifdef SHOW_TOPVIEW
		sprintf_s(resultOutputFileName, "%s_topView.avi", resultFileDate);
		this->m_vwOutputVideo_topView = cvCreateVideoWriter(resultOutputFileName, CV_FOURCC('M','J','P','G'), 15, cvGetSize(currentTopFrame), 1);
#endif
		this->m_bOutputVideoInit = true;
	}	
	cvWriteFrame(this->m_vwOutputVideo, currentFrame);
	delete currentFrame;
#ifdef SHOW_TOPVIEW
	cvWriteFrame(this->m_vwOutputVideo_topView, currentTopFrame);
	delete currentTopFrame;
	topViewResult.release();
#endif
#endif

	displayMat.release();
	tileInputArray.clear();
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
bool CPSNWhere::ReadProjectionSensitivity(cv::Mat &matSensitivity, unsigned int camIdx)
{
	char strFilePath[128];
	sprintf_s(strFilePath, "%s%sProjectionSensitivity_View%03d.txt", 
		this->m_strDatasetPath, 
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
bool CPSNWhere::ReadDistanceFromBoundary(cv::Mat &matDistance, unsigned int camIdx)
{
	char strFilePath[128];
	sprintf_s(strFilePath, "%s%sDistanceFromBoundary_View%03d.txt", 
		this->m_strDatasetPath, 
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

//()()
//('')HAANJU.YOO
