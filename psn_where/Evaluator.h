#include "PSNWhere_Types.h"
#pragma once

typedef std::pair<unsigned int, PSN_Point3D> pointInfo;
typedef std::deque<pointInfo> pointInfoSet;

struct stEvaluationResult
{
	double fMOTA;
	double fMOTP;	
	double fMOTAL;
	double fRecall;
	double fPrecision;
	double fMissTargetPerGroundTruth;
	double fFalseAlarmPerGroundTruth;
	double fFalseAlarmPerFrame;
	int nMissed;
	int nFalsePositives;
	int nIDSwitch;
	int nMostTracked;
	int nPartilalyTracked;
	int nMostLost;	
	int nFragments;
};

class CEvaluator
{
public:
	CEvaluator(void);
	~CEvaluator(void);

	void Initialize(DATASET_TYPE datasetType);
	void Finalize(void);

	void SetResult(PSN_TrackSet &trackSet, unsigned int timeIdx);
	void LoadResultFromText(std::string strFilepath);
	void Evaluate(void);
	//stEvaluationResult EvaluateWithCrop(double cropMargin);

	stEvaluationResult* GetEvaluationResult(void) { return &m_stResult; }
	void PrintResultToConsole(AlgorithmParams *parameters = NULL);
	void PrintResultToFile(const char *strFilepathAndName, AlgorithmParams *parameters = NULL);
	void PrintResultMatrix(const char *strFilepathAndName);
	std::string PrintResultToString(AlgorithmParams *parameters = NULL);

private:
	bool bInit;
	int m_nNumObj;
	int m_nNumTime;
	int m_nSavedResult;

	cv::Mat matXgt;
	cv::Mat matYgt;
	cv::Mat matX;
	cv::Mat matY;

	PSN_Rect m_rectCropZone;
	PSN_Rect m_rectCropZoneMargin;
	PSN_Rect m_rectInnerCropZone;

	std::deque<unsigned int> m_queueID;
	std::vector<pointInfoSet> m_queueSavedResult;

	stEvaluationResult m_stResult;
};

//()()
//('')HAANJU.YOO


