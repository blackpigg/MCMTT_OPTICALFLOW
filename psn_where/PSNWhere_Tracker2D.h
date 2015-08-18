/******************************************************************************
 * Name : PSNWhere_Tracker2D
 * Date : 2014.03.01
 * Author : HAANJU.YOO
 * Version : 0.9
 * Description :
 *	Object detection 결과를 이용하여 2차원 tracklet 생성
 *  [초기화]
 *		-
 *  [주요기능]
 *		-
 *		-
 *	[종료]
 *		-
 ******************************************************************************
 *                                            ....
 *                                           W$$$$$u
 *                                           $$$$F**+           .oW$$$eu
 *                                           ..ueeeWeeo..      e$$$$$$$$$
 *                                       .eW$$$$$$$$$$$$$$$b- d$$$$$$$$$$W
 *                           ,,,,,,,uee$$$$$$$$$$$$$$$$$$$$$ H$$$$$$$$$$$~
 *                        :eoC$$$$$$$$$$$C""?$$$$$$$$$$$$$$$ T$$$$$$$$$$"
 *                         $$$*$$$$$$$$$$$$$e "$$$$$$$$$$$$$$i$$$$$$$$F"
 *                         ?f"!?$$$$$$$$$$$$$$ud$$$$$$$$$$$$$$$$$$$$*Co
 *                         $   o$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 *                 !!!!m.*eeeW$$$$$$$$$$$f?$$$$$$$$$$$$$$$$$$$$$$$$$$$$$U
 *                 !!!!!! !$$$$$$$$$$$$$$  T$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 *                  *!!*.o$$$$$$$$$$$$$$$e,d$$$$$$$$$$$$$$$$$$$$$$$$$$$$$:
 *                 "eee$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$C
 *                b ?$$$$$$$$$$$$$$**$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$!
 *                Tb "$$$$$$$$$$$$$$*uL"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
 *                 $$o."?$$$$$$$$F" u$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 *                  $$$$en ```    .e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
 *                   $$$B*  =*"?.e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$F
 *                    $$$W"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
 *                     "$$$o#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
 *                    R: ?$$$W$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" :!i.
 *                     !!n.?$???""``.......,``````"""""""""""``   ...+!!!
 *                      !* ,+::!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*`
 *                      "!?!!!!!!!!!!!!!!!!!!~ !!!!!!!!!!!!!!!!!!!~`
 *                      +!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!?!`
 *                    .!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!, !!!!
 *                   :!!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!!! `!!:
 *                .+!!!!!!!!!!!!!!!!!!!!!~~!! !!!!!!!!!!!!!!!!!! !!!.
 *               :!!!!!!!!!!!!!!!!!!!!!!!!!.`:!!!!!!!!!!!!!!!!!:: `!!+
 *               "~!!!!!!!!!!!!!!!!!!!!!!!!!!.~!!!!!!!!!!!!!!!!!!!!.`!!:
 *                   ~~!!!!!!!!!!!!!!!!!!!!!!! ;!!!!~` ..eeeeeeo.`+!.!!!!.
 *                 :..    `+~!!!!!!!!!!!!!!!!! :!;`.e$$$$$$$$$$$$$u .
 *                 $$$$$$beeeu..  `````~+~~~~~" ` !$$$$$$$$$$$$$$$$ $b
 *                 $$$$$$$$$$$$$$$$$$$$$UU$U$$$$$ ~$$$$$$$$$$$$$$$$ $$o
 *                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$. $$$$$$$$$$$$$$$~ $$$u
 *                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$$$ 8$$$$.
 *                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$X $$$$$$$$$$$$$$`u$$$$$W
 *                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$".$$$$$$$:
 *                 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  $$$$$$$$$$$$F.$$$$$$$$$
 *                 ?$$$$$$$$$$$$$$$$$$$$$$$$$$$$f $$$$$$$$$$$$' $$$$$$$$$$.
 *                  $$$$$$$$$$$$$$$$$$$$$$$$$$$$ $$$$$$$$$$$$$  $$$$$$$$$$!
 *                  "$$$$$$$$$$$$$$$$$$$$$$$$$$$ ?$$$$$$$$$$$$  $$$$$$$$$$!
 *                   "$$$$$$$$$$$$$$$$$$$$$$$$Fib ?$$$$$$$$$$$b ?$$$$$$$$$
 *                     "$$$$$$$$$$$$$$$$$$$$"o$$$b."$$$$$$$$$$$  $$$$$$$$'
 *                    e. ?$$$$$$$$$$$$$$$$$ d$$$$$$o."?$$$$$$$$H $$$$$$$'
 *                   $$$W.`?$$$$$$$$$$$$$$$ $$$$$$$$$e. "??$$$f .$$$$$$'
 *                  d$$$$$$o "?$$$$$$$$$$$$ $$$$$$$$$$$$$eeeeee$$$$$$$"
 *                  $$$$$$$$$bu "?$$$$$$$$$ 3$$$$$$$$$$$$$$$$$$$$*$$"
 *                 d$$$$$$$$$$$$$e. "?$$$$$:`$$$$$$$$$$$$$$$$$$$$8
 *         e$$e.   $$$$$$$$$$$$$$$$$$+  "??f "$$$$$$$$$$$$$$$$$$$$c
 *        $$$$$$$o $$$$$$$$$$$$$$$F"          `$$$$$$$$$$$$$$$$$$$$b.
 *       M$$$$$$$$U$$$$$$$$$$$$$F"              ?$$$$$$$$$$$$$$$$$$$$$u
 *       ?$$$$$$$$$$$$$$$$$$$$F                   "?$$$$$$$$$$$$$$$$$$$$u
 *        "$$$$$$$$$$$$$$$$$$"                       ?$$$$$$$$$$$$$$$$$$$$o
 *          "?$$$$$$$$$$$$$F                            "?$$$$$$$$$$$$$$$$$$
 *             "??$$$$$$$F                                 ""?3$$$$$$$$$$$$F
 *                                                       .e$$$$$$$$$$$$$$$$'
 *                                                      u$$$$$$$$$$$$$$$$$
 *                                                     `$$$$$$$$$$$$$$$$"
 *                                                      "$$$$$$$$$$$$F"
 *                                                        ""?????""
 *
 ******************************************************************************/

#pragma once

#include "stdafx.h"
#include "PSNWhere_Utils.h"
#include "opencv2/features2d/features2d.hpp"
#include <deque>
#include <list>

// display for 2D tracking
//#define PSN_2D_DEBUG_DISPLAY_

/////////////////////////////////////////////////////////////////////////
// TYPEDEFS
/////////////////////////////////////////////////////////////////////////
struct stDetectedObject
{
	unsigned int id;	
	stDetection detection;
	bool bMatchedWithTracker;
	bool bOverlapWithOtherDetection;
	std::vector<std::vector<cv::Point2f>> vecvecTrackedFeatures; // current -> past order
	std::vector<PSN_Rect> boxes; // current -> past order
	cv::Mat patchGray;
	cv::Mat patchRGB;
	PSN_Point3D location;
	double height;
};

struct stTracker2D
{
	unsigned int id;
	unsigned int timeStart;
	unsigned int timeEnd;
	unsigned int timeLastUpdate;
	unsigned int duration;
	unsigned int numStatic;
	double confidence;
	std::deque<PSN_Rect> boxes;
	std::deque<PSN_Rect> heads;
	std::vector<cv::Point2f> featurePoints;
	std::vector<cv::Point2f> trackedPoints;
	PSN_Point3D lastPosition;
	double height;

	//cv::Mat appModel;
};


/////////////////////////////////////////////////////////////////////////
// CLASS DECLARATION
/////////////////////////////////////////////////////////////////////////
class CPSNWhere_Tracker2D
{
public:
	CPSNWhere_Tracker2D(void);
	~CPSNWhere_Tracker2D(void);

	void Initialize(unsigned int nCamID, stCalibrationInfo *stCalibInfo = NULL);
	void Finalize(void);
	stTrack2DResult& Run(std::vector<stDetection> curDetectionResult, cv::Mat *curFrame, unsigned int frameIdx);

private:
	//----------------------------------------------------------------
	// tracking related
	//----------------------------------------------------------------
	std::vector<cv::Point2f> FindInlierFeatures(std::vector<cv::Point2f> *vecInputFeatures, std::vector<cv::Point2f> *vecOutputFeatures, std::vector<unsigned char> *vecPointStatus);
	static PSN_Rect LocalSearchKLT(PSN_Rect preBox, cv::vector<cv::Point2f> &preFeatures, cv::vector<cv::Point2f> &curFeatures, cv::vector<size_t> &inlierFeatureIndex);

	//double NSSD(cv::Mat *matPatch1, cv::Mat *matPatch2);
	static double BoxMatchingCost(PSN_Rect &box1, PSN_Rect &box2);
	static double GetTrackingConfidence(PSN_Rect &box, std::vector<cv::Point2f> &vecTrackedFeatures);
	PSN_Point2D MotionEstimation(stTracker2D &tracker);

	size_t Track2D_BackwardFeatureTracking(std::vector<stDetection> &curDetectionResult);
	std::vector<float> Track2D_ForwardTrackingAndGetMatchingScore(void);
	void Track2D_MatchingAndUpdating(std::vector<float> &matchingCostArray);

	//----------------------------------------------------------------
	// 3D related
	//----------------------------------------------------------------
	double EstimateDetectionHeight(PSN_Point2D bottomCenter, PSN_Point2D topCenter, PSN_Point3D *location3D = NULL);

	//----------------------------------------------------------------
	// ETC
	//----------------------------------------------------------------
	void ResultWithTracker(stTracker2D *curTracker, stObject2DInfo &outObjectInfo);
	void FilePrintTracklet(void);
	void SaveSnapshot(const char *strFilepath);
	bool LoadSnapshot(const char *strFilepath);	

	/////////////////////////////////////////////////////////////////////
	// VARIABLES
	/////////////////////////////////////////////////////////////////////
	bool m_bInit;
	bool m_bSnapshotLoaded;
	unsigned int m_nCamID;
	unsigned int m_nCurrentFrameIdx;

	// calibration related
	stCalibrationInfo m_stCalibrationInfos;
	unsigned int m_nInputWidth;
	unsigned int m_nInputHeight;
	
	// input related
	std::vector<stDetectedObject> m_vecDetection2D;
	std::vector<cv::Mat*> m_vecPtGrayFrameBuffer;	// FIFO buffer

	// result related
	std::list<stTracker2D> m_listTracker2D;
	std::deque<stTracker2D*> m_queueActiveTracker2D;
	stTrack2DResult m_stTrack2DResult;

	// feature point tracking related
	cv::Ptr<cv::FeatureDetector> m_detector;
	cv::Mat m_matMaskForFeature;
	cv::TermCriteria m_termCriteria;

	// tracker related
	unsigned int m_nNewTrackerID;
	//cv::Mat m_matMaskForPatch;

	// DEBUG
	cv::Mat m_matDebugDisplay;
	char m_strWindowName[128];
	std::deque<double> m_queueOpticalFlowTime;
	std::deque<int> m_queueNumFeatures;
	std::deque<int> m_queueNumOpticalFlow;

#ifdef PSN_2D_DEBUG_DISPLAY_
	//------------
	// DEBUG
	//------------
	std::vector<cv::Scalar> m_vecColors;
	CvVideoWriter *m_vwOutputVideo;
	bool m_bOutputVideoInit;
#endif
};

