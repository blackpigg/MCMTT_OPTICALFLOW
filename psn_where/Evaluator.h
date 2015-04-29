#include "PSNWhere_Manager.h"
#pragma once

typedef std::pair<unsigned int, PSN_Point3D> pointInfo;
typedef std::deque<pointInfo> pointInfoSet;

class CEvaluator
{
public:
	CEvaluator(void);
	~CEvaluator(void);

	void Initialize(std::string strFilepath);
	void Finalize(void);

	void SetResult(PSN_TrackSet &trackSet, unsigned int timeIdx);
	void LoadResultFromText(std::string strFilepath);
	void Evaluate(void);

	void PrintResultToConsole();
	void PrintResultToFile(const char *strFilepathAndName);
	void PrintResultMatrix(const char *strFilepathAndName);

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

	std::deque<unsigned int> m_queueID;
	std::vector<pointInfoSet> m_queueSavedResult;

	double m_fMOTA;
	double m_fMOTP;	
	double m_fMOTAL;
	double m_fRecall;
	double m_fPrecision;
	double m_fMissTargetPerGroundTruth;
	double m_fFalseAlarmPerGroundTruth;
	double m_fFalseAlarmPerFrame;
	int m_nMissed;
	int m_nFalsePositives;
	int m_nIDSwitch;
	int m_nMostTracked;
	int m_nPartilalyTracked;
	int m_nMostLost;	
	int m_nFragments;
};

