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
//#include "Evaluator.h"

/////////////////////////////////////////////////////////////////////////
// DEFINES
/////////////////////////////////////////////////////////////////////////

struct stGraphSolution
{
	PSN_TrackSet tracks;
	double logLikelihood; // = weight sum
	double probability;
	//double weightSum;
	bool bValid;
};

struct stGlobalHypothesis
{
	PSN_TrackSet relatedTracks;
	PSN_TrackSet previousSolution;
};

typedef std::pair<unsigned int, unsigned int> PAIR_UINT;

/////////////////////////////////////////////////////////////////////////
// CLASS DECLARATION
/////////////////////////////////////////////////////////////////////////
class CPSNWhere_Associator3D
{
public:
	CPSNWhere_Associator3D(void);
	~CPSNWhere_Associator3D(void);

	//////////////////////////////////////////////////////////////////////////
	// VARIABLES
	//////////////////////////////////////////////////////////////////////////
public:

private:
	bool m_bInit;
	Etiseo::CameraModel m_cCamModel[NUM_CAM];
	cv::Mat m_matProjectionSensitivity[NUM_CAM];
	cv::Mat m_matDistanceFromBoundary[NUM_CAM];

	char m_strDatasetPath[128];

	// frame related
	cv::Mat *m_matCurrentFrames[NUM_CAM];
	unsigned int m_nCurrentFrameIdx;
	unsigned int m_nNumFramesForProc;
	unsigned int m_nCountForPenalty;
	unsigned int m_nLastDeferredResultPrintFrameIdx;
	unsigned int m_nLastInstantResultPrintFrameIdx;
	double m_fCurrentProcessingTime;
	double m_fCurrentSolvingTime;

	// logging related
	char m_strLogFileName[128];
	char m_strTrackLogFileName[128];
	char m_strResultLogFileName[128];
	char m_strDefferedResultFileName[128];
	char m_strInstantResultFileName[128];
	
	//----------------------------------------------------------------
	// 2D tracklet related
	//----------------------------------------------------------------
	stTracklet2DSet m_vecTracklet2DSet[NUM_CAM];
	unsigned int m_nNumTotalActive2DTracklet;

	//----------------------------------------------------------------
	// 3D track related
	//----------------------------------------------------------------
	bool m_bHasNewMeasurements;
	bool m_bPenaltyFree;
	unsigned int m_nNextTrackID;
	unsigned int m_nNextTreeID;
	unsigned int m_nNextCFamilyID;
	unsigned int m_nNextHypothesisID;
	std::list<TrackTree> m_listTrackTree;
	std::list<Track3D> m_listTrack3D;

	PSN_TrackSet m_queueActiveTrack;
	PSN_TrackSet m_queueDeactivatedTrack;
	PSN_TrackSet m_queueTracksInWindow;
	PSN_TrackSet m_queueTracksInBestSolution;

	std::deque<TrackTree*> m_queueActiveTrees;
	std::deque<TrackTree*> m_queueUnconfirmedTrees;
	std::list<Track3D> m_listResultTrack3D;

	// for debug
	std::deque<Track3D> m_queueForDebugTracksInBestSolution;
	std::deque<stTrack3DResult> m_queueTrackingResult;
	std::deque<stTrack3DResult> m_queueDeferredTrackingResult;
	
	// optimization related
	CGraphSolver m_cGraphSolver;
	std::deque<stGraphSolution> m_stGraphSolutions;
	std::deque<stGlobalHypothesis> m_queueGlobalHypotheses;

	// for visualization
	unsigned int m_nNextVisualizationID;
	std::deque<PAIR_UINT> m_pairTreeIDVisualizationID;	

	//// evaluation
	//CEvaluator m_cEvaluator;
	//CEvaluator m_cEvaluator_Instance;

	//////////////////////////////////////////////////////////////////////////
	// METHODS
	//////////////////////////////////////////////////////////////////////////
public:
	void Initialize(std::string datasetPath, std::vector<stCalibrationInfo*> &vecStCalibInfo);
	void Finalize(void);
	stTrack3DResult Run(std::vector<stTrack2DResult> &curTrack2DResult, cv::Mat *curFrame, int frameIdx);

	// Visualization related
	std::vector<PSN_Point2D> GetHuman3DBox(PSN_Point3D ptHeadCenter, double bodyWidth, unsigned int camIdx);

private:
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

	//----------------------------------------------------------------
	// 3D track related
	//----------------------------------------------------------------		
	void Track3D_UpdateTracks(void);
	void Track3D_GenerateSeedTracks(PSN_TrackSet &outputSeedTracks);
	void Track3D_BranchTracks(PSN_TrackSet &seedTracks);
	void Track3D_UpdateHypotheses(void);
	void Track3D_SolveMHT(void);
	void Track3D_SolveHOMHT(void);
	void Track3D_Pruning_GTP(void);
	void Track3D_Pruning_KBest(void);
	void Track3D_Pruning_Classical(void);
	void Track3D_Pruning_Classical_GTP(void);
	void Track3D_RepairDataStructure(void);

	// cost calculation
	static double ComputeHeadProbability(double distance);
	static double ComputeBodyProbability(double distance, double maxDistance);
	static double ComputeMoveProbability(double distance);
	double ComputeEnterProbability(std::vector<PSN_Point2D_CamIdx> &vecPointInfos);
	double ComputeExitProbability(std::vector<PSN_Point2D_CamIdx> &vecPointInfos);
	static double ComputeLinkProbability(PSN_Point3D &prePoint, PSN_Point3D &curPoint, unsigned int timeGap);

	// miscellaneous
	static bool CheckIncompatibility(Track3D *track1, Track3D *track2);
	static bool CheckIncompatibility(CTrackletCombination &combi1, CTrackletCombination &combi2);
	void GenerateCombinations(std::vector<bool> *vecBAssociationMap, CTrackletCombination combination, std::deque<CTrackletCombination> &combinationQueue, unsigned int camIdx);
	
	//----------------------------------------------------------------
	// Hypothesis related
	//----------------------------------------------------------------	
	PSN_TrackSet Hypothesis_FullGraphCandidateTracks(void);
	void Hypothesis_SolveHOMHT(PSN_TrackSet &tracks, PSN_TrackSet *initialSolutionTracks = NULL);

	//----------------------------------------------------------------
	// interface
	//----------------------------------------------------------------
	stTrack3DResult ResultWithCandidateTracks(void);
	stTrack3DResult ResultWithCurrentBestSolution(void);
	void PrintTracks(std::deque<Track3D*> &queueTracks, char *strFilePathAndName, bool bAppend);
	void FilePrintCurrentTrackTrees(const char *strFilePath);
	void FilePrintDefferedResult(void);
	void FilePrintInstantResult(void);
	void SaveDefferedResult(unsigned int deferredLength);
	void SaveInstantResult(void);

	//----------------------------------------------------------------
	// ETC
	//----------------------------------------------------------------
	static std::deque<std::vector<unsigned int>> IndexCombination(std::deque<std::deque<unsigned int>> &inputIndexDoubleArray, size_t curLevel, std::deque<std::vector<unsigned int>> curCombination);
};

//()()
//('')HAANJU.YOO

