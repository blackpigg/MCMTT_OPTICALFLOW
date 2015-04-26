/******************************************************************************
 * Name : PSNWhere_Associator3D
 * Date : 2014.01.22
 * Author : HAANJU.YOO
 * Version : 0.9
 * Description :
 *	3차원 추적 결과를 생성. 2차원 추적 결과물들을 across view association
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

#include "PSNWhere_Manager.h"
#include "GraphSolver.h"
#include "Evaluator.h"

/////////////////////////////////////////////////////////////////////////
// DEFINES
/////////////////////////////////////////////////////////////////////////

//struct stGraphSolution
//{
//	PSN_TrackSet tracks;
//	double logLikelihood; // = weight sum
//	double probability;
//	//double weightSum;
//	bool bValid;
//};

struct stGlobalHypothesis
{
	PSN_TrackSet selectedTracks;
	PSN_TrackSet relatedTracks;
	double logLikelihood; // = weight sum
	double probability;
	bool bValid;
};
typedef std::deque<stGlobalHypothesis> PSN_HypothesisSet;

typedef std::pair<unsigned int, unsigned int> PAIR_UINT;

/////////////////////////////////////////////////////////////////////////
// CLASS DECLARATION
/////////////////////////////////////////////////////////////////////////
class CPSNWhere_Associator3D
{
public:
	CPSNWhere_Associator3D(void);
	~CPSNWhere_Associator3D(void);
	void Initialize(std::string datasetPath, std::vector<stCalibrationInfo*> &vecStCalibInfo);
	void Finalize(void);
	stTrack3DResult Run(std::vector<stTrack2DResult> &curTrack2DResult, cv::Mat *curFrame, int frameIdx);

	// Visualization related
	std::vector<PSN_Point2D> GetHuman3DBox(PSN_Point3D ptHeadCenter, double bodyWidth, unsigned int camIdx);

private:
	//////////////////////////////////////////////////////////////////////////
	// METHODS
	//////////////////////////////////////////////////////////////////////////

	//----------------------------------------------------------------
	// 3D geometry related
	//----------------------------------------------------------------
	PSN_Point2D WorldToImage(PSN_Point3D point3D, int camIdx);
	PSN_Point3D ImageToWorld(PSN_Point2D point2D, double z, int camIdx);
	bool ReadProjectionSensitivity(cv::Mat &matSensitivity, unsigned int camIdx);
	bool ReadDistanceFromBoundary(cv::Mat &matDistance, unsigned int camIdx);

	bool CheckVisibility(PSN_Point3D testPoint, unsigned int camIdx);
	bool CheckHeadWidth(PSN_Point3D midPoint3D, PSN_Rect rect1, PSN_Rect rect2, unsigned int camIdx1, unsigned int camIdx2);

	stReconstruction PointReconstruction(CTrackletCombination &tracklet2Ds);
	double NViewPointReconstruction(std::vector<PSN_Line> &vecLines, PSN_Point3D &outputPoint);
	double NViewGroundingPointReconstruction(std::vector<PSN_Point2D_CamIdx> &vecPointInfos, PSN_Point3D &outputPoint);
	PSN_Line GetBackProjectionLine(PSN_Point2D point2D, unsigned int camIdx);	
	
	//----------------------------------------------------------------
	// 2D tracklet related
	//----------------------------------------------------------------
	void Tracklet2D_UpdateTracklets(std::vector<stTrack2DResult> &curTrack2DResult, unsigned int frameIdx);	
	void GenerateTrackletCombinations(std::vector<bool> *vecBAssociationMap, CTrackletCombination combination, std::deque<CTrackletCombination> &combinationQueue, unsigned int camIdx);

	//----------------------------------------------------------------
	// 3D track related
	//----------------------------------------------------------------	
	void Track3D_Management(PSN_TrackSet &outputSeedTracks);
	void Track3D_UpdateTracks(void);
	void Track3D_GenerateSeedTracks(PSN_TrackSet &outputSeedTracks);
	void Track3D_BranchTracks(PSN_TrackSet *seedTracks);
	PSN_TrackSet Track3D_GetWholeCandidateTracks(void);

	//void Track3D_SolveHOMHT(void);

	//void Track3D_Pruning_GTP(void);
	//void Track3D_Pruning_KBest(void);
	//void Track3D_RepairDataStructure(void);

	// cost calculation
	double ComputeEnterProbability(std::vector<PSN_Point2D_CamIdx> &vecPointInfos);
	double ComputeExitProbability(std::vector<PSN_Point2D_CamIdx> &vecPointInfos);
	static double ComputeLinkProbability(PSN_Point3D &prePoint, PSN_Point3D &curPoint, double preSensitivity, double curSensitivity, unsigned int timeGap);
	double ComputeRGBCost(const cv::Mat *feature1, const cv::Mat *feature2, unsigned int timeGap);

	// miscellaneous
	static bool CheckIncompatibility(Track3D *track1, Track3D *track2);
	static bool CheckIncompatibility(CTrackletCombination &combi1, CTrackletCombination &combi2);
	cv::Mat GetRGBFeature(const cv::Mat *patch, int numBins);
	
	//----------------------------------------------------------------
	// Hypothesis related
	//----------------------------------------------------------------	
	void Hypothesis_UpdateHypotheses(PSN_HypothesisSet &inoutUpdatedHypotheses, PSN_TrackSet *newSeedTracks);
	void Hypothesis_Formation(PSN_HypothesisSet &outBranchHypotheses, PSN_HypothesisSet *existingHypotheses);
	void Hypothesis_BranchHypotheses(PSN_HypothesisSet &outBranchHypotheses, PSN_TrackSet *tracks, PSN_TrackSet *initialSolutionTracks = NULL);
	void Hypothesis_PruningNScanBack(unsigned int nCurrentFrameIdx, unsigned int N, PSN_TrackSet *tracksInWindow, std::deque<stGlobalHypothesis> *ptQueueHypothesis = NULL);
	void Hypothesis_PruningTrackWithGTP(unsigned int nCurrentFrameIdx, unsigned int nNumMaximumTrack, PSN_TrackSet *tracksInWindow);
	void Hypothesis_RefreshHypotheses(PSN_HypothesisSet &inoutUpdatedHypotheses);

	//----------------------------------------------------------------
	// interface
	//----------------------------------------------------------------
	stTrack3DResult ResultWithTracks(PSN_TrackSet *trackSet, unsigned int nFrameIdx, double fProcessingTime = 0);
	void PrintTracks(std::deque<Track3D*> &queueTracks, char *strFilePathAndName, bool bAppend);
	void PrintHypotheses(PSN_HypothesisSet &queueHypotheses, char *strFilePathAndName, unsigned int frameIdx);
	void PrintCurrentTrackTrees(const char *strFilePath);
	void PrintResult(const char *strFilepath, std::deque<stTrack3DResult> *queueResults);
//	void FilePrintInstantResult(void);
//	void SaveDefferedResult(unsigned int deferredLength);
//	void SaveInstantResult(void);

	//----------------------------------------------------------------
	// ETC
	//----------------------------------------------------------------
	static std::deque<std::vector<unsigned int>> IndexCombination(std::deque<std::deque<unsigned int>> &inputIndexDoubleArray, size_t curLevel, std::deque<std::vector<unsigned int>> curCombination);
	void SaveSnapshot(const char *strFilepath);
	bool LoadSnapshot(const char *strFilepath);

	//////////////////////////////////////////////////////////////////////////
	// VARIABLES
	//////////////////////////////////////////////////////////////////////////
	bool bInit_;
	bool bSnapshotReaded_;
	Etiseo::CameraModel cCamModel_[NUM_CAM];
	cv::Mat matProjectionSensitivity_[NUM_CAM];
	cv::Mat matDistanceFromBoundary_[NUM_CAM];

	char strDatasetPath_[128];

	// frame related
	cv::Mat *ptMatCurrentFrames_[NUM_CAM];
	unsigned int nCurrentFrameIdx_;
	unsigned int nNumFramesForProc_;
	unsigned int nCountForPenalty_;
	unsigned int nLastPrintedDeferredResultFrameIdx_;
	unsigned int nLastPrintedInstantResultFrameIdx_;
	double fCurrentProcessingTime_;
	double fCurrentSolvingTime_;

	// logging related
	char strLogFileName_[128];
	char strTrackLogFileName_[128];
	char strResultLogFileName_[128];
	char strDefferedResultFileName_[128];
	char strInstantResultFileName_[128];
	
	//----------------------------------------------------------------
	// 2D tracklet related
	//----------------------------------------------------------------
	stTracklet2DSet vecTracklet2DSet_[NUM_CAM];
	unsigned int nNumTotalActive2DTracklet_;

	//----------------------------------------------------------------
	// 3D track related
	//----------------------------------------------------------------
	bool bReceiveNewMeasurement_;
	bool bInitiationPenaltyFree_;
	unsigned int nNewTrackID_;
	unsigned int nNewTreeID_;
	std::list<TrackTree> listTrackTree_;
	std::list<Track3D> listTrack3D_;

	PSN_TrackSet queueNewSeedTracks_;
	PSN_TrackSet queueActiveTrack_;
	PSN_TrackSet queuePausedTrack_;
	PSN_TrackSet queueTracksInWindow_;
	PSN_TrackSet queueTracksInBestSolution_;

	std::deque<TrackTree*> queuePtActiveTrees_;
	std::deque<TrackTree*> queuePtUnconfirmedTrees_;

	// for result saving
	std::deque<stTrack3DResult> queueTrackingResult_;
	std::deque<stTrack3DResult> queueDeferredTrackingResult_;
	
	// optimization related
	CGraphSolver cGraphSolver_;
	//std::deque<stGraphSolution> queueStGraphSolutions_;
	PSN_HypothesisSet queuePrevGlobalHypotheses_;
	PSN_HypothesisSet queueCurrGlobalHypotheses_;

	// for visualization
	unsigned int nNewVisualizationID_;
	std::deque<PAIR_UINT> queuePairTreeIDToVisualizationID_;	

	// evaluation
	CEvaluator m_cEvaluator;
	CEvaluator m_cEvaluator_Instance;

	// for debugging
	int nCountTrackInOptimization_;
	int nCountUCTrackInOptimization_;
};

//()()
//('')HAANJU.YOO

