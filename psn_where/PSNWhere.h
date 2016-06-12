#pragma once

#include "PSNWhere_Utils.h"
#include "Evaluator.h"

class CPSNWhere_Tracker2D;
class CPSNWhere_Associator3D;

/////////////////////////////////////////////////////////////////////
// API CLASS
/////////////////////////////////////////////////////////////////////
class CPSNWhere
{	
	///////////////////////////////////////////
	// METHODS
	///////////////////////////////////////////
public:
	// constructor and destructor
	CPSNWhere(void);
	~CPSNWhere(void);

	// init & destroy
	bool Initialize(DATASET_TYPE datasetType, AlgorithmParams *parameters = NULL);
	void Finalize();

	// tracking
	TrackInfoArray* TrackPeople(std::vector<cv::Mat> inputFrames, int frameIdx);

	// display
	void Visualize(std::vector<cv::Mat> inputFrames, int frameIdx, std::vector<stTrack2DResult> &result2D, stTrack3DResult &result3D, int nDispMode);

	// calibration
	bool ReadProjectionSensitivity(cv::Mat &matSensitivity, unsigned int camIdx);
	bool ReadDistanceFromBoundary(cv::Mat &matDistance, unsigned int camIdx);

	///////////////////////////////////////////
	// VARIABLES
	///////////////////////////////////////////
private:
	bool m_bInit;
	char m_strDatasetPath[128];
	std::string m_strTime;
	AlgorithmParams m_parameters;
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

	// evaluation
	std::vector<std::pair<CEvaluator, int>> m_vecEvaluator;
};


//()()
//('')HAANJU.YOO
