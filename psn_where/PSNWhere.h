#pragma once

#include "PSNWhere_Manager.h"

class CPSNWhere_Tracker2D;
class CPSNWhere_Associator3D;

/////////////////////////////////////////////////////////////////////
// API CLASS
/////////////////////////////////////////////////////////////////////
class CPSNWhere
{
	///////////////////////////////////////////
	// VARIABLES
	///////////////////////////////////////////
public:

private:
	bool m_bInit;
	char m_strDatasetPath[128];
	CPSNWhere_Tracker2D *m_cTracker2D[NUM_CAM];
	CPSNWhere_Associator3D *m_cAssociator3D;

	stCalibrationInfo m_stCalibrationInfos[NUM_CAM];

	// output related
	cv::Mat m_matResultFrames[NUM_CAM];
	cv::Mat m_matTopViewBase;	
	std::vector<cv::Scalar> m_vecColors;
	CvVideoWriter *m_vwOutputVideo;
	CvVideoWriter *m_vwOutputVideo_topView;
	bool m_bOutputVideoInit;
	double m_fProcessingTime;

	///////////////////////////////////////////
	// METHODS
	///////////////////////////////////////////
public:
	// constructor and destructor
	CPSNWhere(void);
	~CPSNWhere(void);

	// init & destroy
	bool Initialize(std::string datasetPath);
	void Finalize();

	// tracking
	TrackInfoArray* TrackPeople(cv::Mat *pDibArray, int frameIdx);

	// display
	void Visualize(cv::Mat *pDibArray, int frameIdx, std::vector<stTrack2DResult> &result2D, stTrack3DResult &result3D, int nDispMode);

	// calibration
	bool ReadProjectionSensitivity(cv::Mat &matSensitivity, unsigned int camIdx);
	bool ReadDistanceFromBoundary(cv::Mat &matDistance, unsigned int camIdx);
private:
};


//()()
//('')HAANJU.YOO
