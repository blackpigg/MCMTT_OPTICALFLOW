#pragma once

#include "stdafx.h"
#include <queue>
#include <time.h>
#include <list>
#include <deque>

#include "cv.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "calibration\cameraModel.h"
#include "PSNWhere_SGSmooth.h"

#define PSN_DEBUG_MODE_
#define PSN_MONITOR_MODE_
//#define PSN_PRINT_LOG_
//#define LOAD_SNAPSHOT_

//#define SAVE_SNAPSHOT_
#define DO_RECORD
#define SHOW_TOPVIEW

/////////////////////////////////////////////////////////////////////////
// PATH
/////////////////////////////////////////////////////////////////////////
//#define DATASET_PATH ("D:\\Workspace\\Dataset\\ETRI\\P5\\")
#define CALIBRATION_PATH ("/calibrationInfos/")
#define DETECTION_PATH ("/detectionResult/")
#define TRACKLET_PATH ("/trackletInput/")
#define RESULT_SAVE_PATH ("D:/Workspace/ExperimentalResult/150124_[C++]_PETS_Result_ICIP/")
#define SNAPSHOT_PATH ("data/")

/////////////////////////////////////////////////////////////////////////
// EXPERIMENTAL PRESETS
/////////////////////////////////////////////////////////////////////////
#define PSN_INPUT_TYPE (1)	// 0:ETRI / 1:PETS2009
#define PSN_DETECTION_TYPE (1)	// 0:Head / 1:Full-body

#if 0 == PSN_INPUT_TYPE
	// ETRI Testbed setting
	define NUM_CAM (4)
	const unsigned int CAM_ID[NUM_CAM] = {0, 1, 2, 3};
	const double CAM_HEIGHT_SCALE[NUM_CAM] = {1.0, 1.0, 1.0, 1.0};
#else
	// PETS.S2.L1 setting
	#define NUM_CAM (3)
	const unsigned int CAM_ID[NUM_CAM] = {1, 5, 7};
	//const double CAM_HEIGHT_SCALE[NUM_CAM] = {1.0, 0.77, 0.77};
	const double CAM_HEIGHT_SCALE[NUM_CAM] = {1.0, 1.0, 1.0};
#endif

#define NUM_DETECTION_PART (8)
const std::string DETCTION_PART_NAME[NUM_DETECTION_PART] = {"HEAD", "F1", "S1", "GR", "S2", "A1", "A2", "F2"};

/////////////////////////////////////////////////////////////////////////
// PREDEFINED VALUES
/////////////////////////////////////////////////////////////////////////
#define PSN_P_INF_SI (INT_MAX)
#define PSN_N_INF_SI (INT_MIN)
#define PSN_P_INF_F (FLT_MAX)
#define PSN_N_INF_F (FLT_MIN)
#define PSN_PI (3.1415926535897);

/////////////////////////////////////////////////////////////////////////
// VISUALIZATION SETTTING
/////////////////////////////////////////////////////////////////////////
#define DISP_TRAJECTORY3D_LENGTH (40)
#define DISPLAY_ID_MODE (0) // 0: raw track id, 1: id for visualization

/////////////////////////////////////////////////////////////////////////
// EVALUATION SETTING
/////////////////////////////////////////////////////////////////////////
#define CROP_ZONE_X_MIN (-14069.6)
#define CROP_ZONE_X_MAX (4981.3)
#define CROP_ZONE_Y_MIN (-14274.0)
#define CROP_ZONE_Y_MAX (1733.5)
#define CROP_ZONE_MARGIN (1000.0)

/////////////////////////////////////////////////////////////////////////
// TYPEDEFS
/////////////////////////////////////////////////////////////////////////
class PSN_Point2D
{
public:

	// data
	double x;
	double y;

	// constructors
	PSN_Point2D() : x(0), y(0) {}
	PSN_Point2D(double x, double y) : x(x), y(y){}
	PSN_Point2D(cv::Point2f a) : x((double)a.x), y((double)a.y){}

	// operators
	PSN_Point2D& operator=(const PSN_Point2D &a){ x = a.x; y = a.y; return *this; }
	PSN_Point2D& operator=(const cv::Point &a){ x = a.x; y = a.y; return *this; }
	PSN_Point2D& operator=(const cv::Point2f &a){ x = (double)a.x; y = (double)a.y; return *this; }
	PSN_Point2D& operator+=(const PSN_Point2D &a){ x = x + a.x; y = y + a.y; return *this; }
	PSN_Point2D& operator-=(const PSN_Point2D &a){ x = x - a.x; y = y - a.y; return *this; }
	PSN_Point2D& operator+=(const double s){ x = x + s; y = y + s; return *this; }
	PSN_Point2D& operator-=(const double s){ x = x - s; y = y - s; return *this; }
	PSN_Point2D& operator*=(const double s){ x = x * s; y = y * s; return *this; }
	PSN_Point2D& operator/=(const double s){ x = x / s; y = y / s; return *this; }
	PSN_Point2D operator+(const PSN_Point2D &a){ return PSN_Point2D(x + a.x, y + a.y); }
	PSN_Point2D operator+(const double s){ return PSN_Point2D(x + s, y + s); }
	PSN_Point2D operator-(const PSN_Point2D &a){ return PSN_Point2D(x - a.x, y - a.y); }
	PSN_Point2D operator-(const double s){ return PSN_Point2D(x - s, y - s); }
	PSN_Point2D operator-(){ return PSN_Point2D(-x, -y); }
	PSN_Point2D operator*(const double s){ return PSN_Point2D(x * s, y * s); }
	PSN_Point2D operator/(const double s){ return PSN_Point2D(x / s, y / s); }
	bool operator==(const PSN_Point2D &a){ return (x == a.x && y == a.y); }
	bool operator==(const cv::Point &a){ return (x == a.x && y == a.y); }

	// methods
	double norm_L2(){ return std::sqrt(x * x + y * y); }
	double dot(const PSN_Point2D &a){ return x * a.x + y * a.y; }
	bool onView(const unsigned int width, const unsigned int height)
	{ 
		if(x < 0){ return false; }
		if(x >= (double)width){ return false; }
		if(y < 0){ return false; }
		if(y >= (double)height){ return false; }
		return true; 
	}
	PSN_Point2D scale(double scale)
	{
		return PSN_Point2D(x * scale, y * scale);
	}

	// data conversion
	cv::Point cv(){ return cv::Point((int)x, (int)y); }
};

class PSN_Point3D
{
public:

	// data
	double x;
	double y;
	double z;

	// constructors
	PSN_Point3D() : x(0), y(0), z(0) {}
	PSN_Point3D(double x, double y, double z) :x(x), y(y), z(z)	{}

	// operators
	PSN_Point3D& operator=(const PSN_Point3D &a){ x = a.x; y = a.y; z = a.z; return *this; }
	PSN_Point3D& operator=(const cv::Point3d &a){ x = a.x; y = a.y; z = a.z; return *this; }
	PSN_Point3D& operator+=(const PSN_Point3D &a){ x = x + a.x; y = y + a.y; z = z + a.z; return *this; }
	PSN_Point3D& operator-=(const PSN_Point3D &a){ x = x - a.x; y = y - a.y; z = z - a.z; return *this; }
	PSN_Point3D& operator+=(const double s){ x = x + s; y = y + s; z = z + s; return *this; }
	PSN_Point3D& operator-=(const double s){ x = x - s; y = y - s; z = z - s; return *this; }
	PSN_Point3D& operator*=(const double s){ x = x * s; y = y * s; z = z * s; return *this; }
	PSN_Point3D& operator/=(const double s){ x = x / s; y = y / s; z = z / s; return *this; }
	PSN_Point3D operator+(const PSN_Point3D &a){ return PSN_Point3D(x + a.x, y + a.y, z + a.z); }
	PSN_Point3D operator+(const double s){ return PSN_Point3D(x + s, y + s, z + s); }
	PSN_Point3D operator-(const PSN_Point3D &a){ return PSN_Point3D(x - a.x, y - a.y, z - a.z); }
	PSN_Point3D operator-(const double s){ return PSN_Point3D(x - s, y - s, z - s); }
	PSN_Point3D operator-(){ return PSN_Point3D(-x, -y, -z); }
	PSN_Point3D operator*(const double s){ return PSN_Point3D(x * s, y * s, z * s); }
	PSN_Point3D operator/(const double s){ return PSN_Point3D(x / s, y / s, z / s); }

	bool operator==(const PSN_Point3D &a){ return (x == a.x && y == a.y && z == a.z); }
	bool operator==(const cv::Point3d &a){ return (x == a.x && y == a.y && z == a.z); }

	// methods
	double norm_L2(){ return std::sqrt(x * x + y * y + z * z); }
	double dot(const PSN_Point3D &a){ return x * a.x + y * a.y + z * a.z; }

	// data conversion
	cv::Point3d cv(){ return cv::Point3d(x, y, z); }
};

class PSN_Rect
{
public:
	double x;
	double y;
	double w;
	double h;

	// constructors
	PSN_Rect() : x(0), y(0), w(0), h(0) {}
	PSN_Rect(double x, double y, double w, double h) :x(x), y(y), w(w), h(h) {}

	// operators
	PSN_Rect& operator=(const PSN_Rect &a){ x = a.x; y = a.y; w = a.w; h = a.h; return *this; }
	PSN_Rect& operator=(const cv::Rect &a){ x = a.x; y = a.y; w = a.width; h = a.height; return *this; }
	bool operator==(const PSN_Rect &a){ return (x == a.x && y == a.y && w == a.w && h == a.h); }
	bool operator==(const cv::Rect &a){ return (x == a.x && y == a.y && w == a.width && h == a.height); }

	// methods	
	PSN_Point2D bottomCenter(){ return PSN_Point2D(x + std::ceil(w/2.0), y + h); }
	PSN_Point2D center(){ return PSN_Point2D(x + std::ceil(w/2.0), y + std::ceil(h/2.0)); }
	PSN_Point2D reconstructionPoint()
	{
		return this->bottomCenter();
		//switch(PSN_INPUT_TYPE)
		//{
		//case 1:
		//	return this->bottomCenter();
		//	break;
		//default:
		//	return this->center();
		//	break;
		//}
	}
	PSN_Rect cropWithSize(const double width, const double height)
	{
		double newX = std::max(0.0, x);
		double newY = std::max(0.0, y);
		double newW = std::min(width - newX - 1, w);
		double newH = std::min(height - newY - 1, h);
		return PSN_Rect(newX, newY, newW, newH);
	}
	PSN_Rect scale(double scale)
	{
		return PSN_Rect(x * scale, y * scale, w * scale, h * scale);
	}
	double area(){ return w * h; }
	bool contain(const PSN_Point2D &a){ return (a.x >= x && a.x < x + w && a.y >= y && a.y < y + h); }
	bool contain(const cv::Point2f &a){ return ((double)a.x >= x && (double)a.x < x + w && (double)a.y >= y && (double)a.y < y + h); }
	bool overlap(const PSN_Rect &a)
	{ 
		return (std::max(x + w, a.x + a.w) - std::min(x, a.x) < w + a.w) && (std::max(y + h, a.y + a.h) - std::min(y, a.y) < h + a.h) ? true : false; 
	}
	double distance(const PSN_Rect &a)
	{ 
		PSN_Point3D descriptor1 = PSN_Point3D(x + w/2.0, y + h/2.0, w);
		PSN_Point3D descriptor2 = PSN_Point3D(a.x + a.w/2.0, a.y + a.h/2.0, a.w);
		return (descriptor1 - descriptor2).norm_L2() / std::min(w, a.w);
	}
	double overlappedArea(const PSN_Rect &a)
	{
		double overlappedWidth = std::min(x + w, a.x + a.w) - std::max(x, a.x);
		if (0.0 >= overlappedWidth) { return 0.0; }
		double overlappedHeight = std::min(y + h, a.y + a.h) - std::max(y, a.y);
		if (0.0 >= overlappedHeight) { return 0.0; }
		return overlappedWidth * overlappedHeight;
	}

	// conversion
	cv::Rect cv(){ return cv::Rect((int)x, (int)y, (int)w, (int)h); }
};

typedef struct _stDetection
{
	PSN_Rect box;
	std::vector<PSN_Rect> vecPartBoxes;
} stDetection;

typedef struct _stObject2DInfo
{
	unsigned int id;
	PSN_Rect box;
	double score;
} stObject2DInfo;

typedef struct _stTrack2DResult
{
	unsigned int camID;
	unsigned int frameIdx;
	std::vector<stObject2DInfo> object2DInfos;

	std::vector<PSN_Rect> vecDetectionRects;
	std::vector<PSN_Rect> vecTrackerRects;
	cv::Mat matMatchingCost;
} stTrack2DResult;

typedef struct _stObject3DInfo
{
	unsigned int id;
	std::vector<PSN_Point3D> recentPoints;
	std::vector<PSN_Point2D> recentPoint2Ds[NUM_CAM];
	std::vector<PSN_Point2D> point3DBox[NUM_CAM];
	std::vector<PSN_Point3D> curDetectionPosition;
	PSN_Rect rectInViews[NUM_CAM];
	bool bVisibleInViews[NUM_CAM];
} stObject3DInfo;

typedef struct _stTrack3DResult
{
	unsigned int frameIdx;
	double processingTime;
	std::vector<stObject3DInfo> object3DInfo;
}stTrack3DResult;


// TrackInfo structure
typedef struct _stTrackInfo
{
	int nTrackID;			// 트래킹 아이디	
	PSN_Point3D point3D;	// 위치

	int nLifeTime;			// 얼마나 오래 트래킹되고 있는가? (단위: frame)
	int nCertainty;			// 트랭킹 결과가 얼마나 믿을만 한가? (제공가능한가?)	

	PSN_Rect rtHead[NUM_CAM];		//각 캠에서 검출된 머리영역
	PSN_Rect rtBody[NUM_CAM];		//각 캠에서 검출된 바디영역
} stTrackInfo;

typedef CArray<stTrackInfo, stTrackInfo&> TrackInfoArray;			// MFC, 한 시점에서 존재하는 모든 물체에 대한 트래킹 정보. 길이 = 물체 개수

// calibration information
typedef struct _stCalibrationInfos
{
	unsigned int nCamIdx;
	Etiseo::CameraModel cCamModel;	
	cv::Mat matProjectionSensitivity;
	cv::Mat matDistanceFromBoundary;
} stCalibrationInfo;

typedef std::pair<PSN_Point3D, PSN_Point3D> PSN_Line;
typedef std::pair<PSN_Point2D, unsigned int> PSN_Point2D_CamIdx;

class TrackTree;
struct stTracklet2D
{
	unsigned int id;
	unsigned int camIdx;
	bool bActivated;

	// spatial information
	std::deque<PSN_Rect> rects;
	std::deque<PSN_Line> backprojectionLines;

	// temporal information
	unsigned int timeStart;
	unsigned int timeEnd;
	unsigned int duration;

	// appearance 
	cv::Mat RGBFeatureHead;
	cv::Mat RGBFeatureTail;

	// location in 3D
	PSN_Point3D currentLocation3D;

	// matching related
	std::vector<bool> bAssociableNewMeasurement[NUM_CAM];
};

struct stTracklet2DSet
{
	std::list<stTracklet2D> tracklets;
	std::deque<stTracklet2D*> activeTracklets; // for fast searching
	std::deque<stTracklet2D*> newMeasurements; // for generating seeds
};

class CTrackletCombination
{
private:
	stTracklet2D *tracklets[NUM_CAM];
	unsigned int numTracklets;

public:
	CTrackletCombination(void);

	// operator
	CTrackletCombination& operator=(const CTrackletCombination &a);
	bool operator==(const CTrackletCombination &a);

	// methods
	void set(unsigned int camIdx, stTracklet2D *tracklet);
	stTracklet2D* get(unsigned int camIdx);
	void print(void);
	bool checkCoupling(CTrackletCombination &a);
	unsigned int size(void);
};

struct stReconstruction
{
	bool bIsMeasurement;
	CTrackletCombination tracklet2Ds;
	std::vector<PSN_Point3D> rawPoints;
	PSN_Point3D point;
	PSN_Point3D smoothedPoint;
	PSN_Point3D velocity;
	double maxError;
	double costReconstruction;
	double costSmoothedPoint;
	double costLink;
};

class CPointSmoother
{
public:
	CPointSmoother(void);
	~CPointSmoother(void);
	// setter
	int Insert(PSN_Point3D &point);
	int Insert(std::vector<PSN_Point3D> &points);
	int ReplaceBack(PSN_Point3D &point);
	void SetQsets(std::vector<Qset> *Qsets);
	void PopBack(void);
	void SetSmoother(std::deque<PSN_Point3D> &data, std::deque<PSN_Point3D> &smoothedPoints, int span, int degree);
	// getter
	PSN_Point3D GetResult(int pos);
	std::vector<PSN_Point3D> GetResults(int startPos, int endPos = -1);
	size_t size(void) const { return size_; }
	PSN_Point3D back(void) const { return *back_; }
	void GetSmoother(std::deque<PSN_Point3D> &data, std::deque<PSN_Point3D> &smoothedPoints, int &span, int &degree);

private:
	void Update(int refreshPos, int numPoints);

	size_t size_;
	PSN_Point3D *back_;
	CPSNWhere_SGSmooth smootherX_, smootherY_, smootherZ_;	
	std::deque<PSN_Point3D> smoothedPoints_;
};

struct stTrackIndexElement;
class Track3D
{
public:
	Track3D();
	~Track3D();

	void Initialize(unsigned int id, Track3D *parentTrack, unsigned int timeGeneration, CTrackletCombination &trackletCombination);
	void RemoveFromTree();
	static std::deque<Track3D*> GatherValidChildrenTracks(Track3D* validParentTrack, std::deque<Track3D*> &targetChildrenTracks);
	//void SetKalmanFilter(PSN_Point3D &initialPoint);

	unsigned int id;
	CTrackletCombination curTracklet2Ds;
	std::deque<unsigned int> tracklet2DIDs[NUM_CAM];	
	bool bActive;		// for update and branching
	bool bValid;		// for deletion

	// for tree
	TrackTree *tree;
	Track3D *parentTrack;
	std::deque<Track3D*> childrenTrack;
	
	// temporal information
	unsigned int timeStart;
	unsigned int timeEnd;
	unsigned int timeGeneration;
	unsigned int duration;
	
	// reconstruction related
	std::deque<stReconstruction> reconstructions;

	// smoothing related
	CPointSmoother smoother;

	// cost
	double costTotal;
	double costReconstruction;
	double costLink;
	double costEnter;
	double costExit;
	double costRGB;

	// loglikelihood
	double loglikelihood;

	// global track probability
	double GTProb;
	double BranchGTProb;
	bool bWasBestSolution;
	bool bCurrentBestSolution;

	// HO-HMT
	bool bNewTrack;	

	// tracklet related	
	unsigned int timeTrackletEnded[NUM_CAM];
	PSN_Point3D lastTrackletLocation3D[NUM_CAM];
	double lastTrackletSensitivity[NUM_CAM];
	cv::Mat lastRGBFeature[NUM_CAM];

	// termination
	unsigned int numOutpoint; // count tpoints
};

typedef std::deque<Track3D*> PSN_TrackSet;

//struct stTracklet2DInfo
//{
//	stTracklet2D *tracklet2D;
//	std::deque<Track3D*> queueRelatedTracks;
//};

struct stTrackInTreeInfo
{
	int id;
	int parentNode;
	int timeGenerated;
	float GTP;
};

class TrackTree
{
public:
	unsigned int id;
	unsigned int timeGeneration;
	bool bValid;
	std::deque<Track3D*> tracks; // seed at the first

	//unsigned int numMeasurements;
	//std::deque<stTracklet2DInfo> tracklet2Ds[NUM_CAM];

	// pruning related
	//double maxGTProb;

public:
	TrackTree();
	~TrackTree();

	void Initialize(unsigned int id, Track3D *seedTrack, unsigned int timeGeneration, std::list<TrackTree> &treeList);
	void ResetGlobalTrackProbInTree();

	Track3D* FindPruningPoint(unsigned int timeWindowStart, Track3D *rootOfBranch = NULL);
	static bool CheckBranchContainsBestSolution(Track3D *rootOfBranch);
	static void SetValidityFlagInTrackBranch(Track3D* rootOfBranch, bool bValid);	
	static void GetTracksInBranch(Track3D* rootOfBranch, std::deque<Track3D*> &queueOutput);
	static void MakeTreeNodesWithChildren(std::deque<Track3D*> queueChildrenTracks, const int parentNodeIdx, std::deque<stTrackInTreeInfo> &outQueueNodes);
	static double GTProbOfBrach(Track3D *rootOfBranch);
	static double MaxGTProbOfBrach(Track3D *rootOfBranch);
	static void InvalidateBranchWithMinGTProb(Track3D *rootOfBranch, double minGTProb);
	static Track3D* FindMaxGTProbBranch(Track3D* branchSeedTrack, size_t timeIndex);
	static Track3D* FindOldestTrackInBranch(Track3D *trackInBranch, int nMostPreviousFrameIdx);
	//static bool CheckConnectivityOfTrees(TrackTree *tree1, TrackTree *tree2);
	

private:
};

struct stTreeCluster
{
	unsigned int id;
	size_t numTrees;
	std::deque<TrackTree*> trees;
};


/////////////////////////////////////////////////////////////////////////
// OPERATOR
/////////////////////////////////////////////////////////////////////////
namespace psn
{

// matrix operation
template<typename _Tp> _Tp MatTotalSum(cv::Mat &inputMat)
{
	_Tp resultSum = 0;
	for(int rowIdx = 0; rowIdx < inputMat.rows; rowIdx++)
	{
		for(int colIdx = 0; colIdx < inputMat.cols; colIdx++)
		{
			resultSum += inputMat.at<_Tp>(rowIdx, colIdx);
		}
	}
	return resultSum;
}

template<typename _Tp> bool MatLowerThan(cv::Mat &inputMat, _Tp compValue)
{
	//_Tp maxValue = std::max_element(inputMat.begin(), inputMat.end());
	//for(int rowIdx = 0; rowIdx < inputMat.rows; rowIdx++)
	//{
	//	for(int colIdx = 0; colIdx < inputMat.cols; colIdx++)
	//	{
	//		if(inputMat.at<_Tp>(rowIdx, colIdx) >= compValue)
	//		{
	//			return false;
	//		}
	//	}
	//}
	return (std::max_element(inputMat.begin(), inputMat.end()) < compValue) ? true : false;
}

template<typename _Tp> bool MatContainLowerThan(cv::Mat &inputMat, _Tp compValue)
{
	for(int rowIdx = 0; rowIdx < inputMat.rows; rowIdx++)
	{
		for(int colIdx = 0; colIdx < inputMat.cols; colIdx++)
		{
			if(inputMat.at<_Tp>(rowIdx, colIdx) < compValue)
			{
				return true;
			}
		}
	}
	return false;
}

template<typename _Tp> std::vector<_Tp> mat2vec_C1(cv::Mat &inputMat)
{
	std::vector<_Tp> vecResult;

	for(int rowIdx = 0; rowIdx < inputMat.rows; rowIdx++)
	{
		for(int colIdx = 0; colIdx < inputMat.cols; colIdx++)
		{
			vecResult.push_back(inputMat.at<_Tp>(rowIdx, colIdx));
		}
	}
	
	return vecResult;
}

void appendRow(cv::Mat &dstMat, cv::Mat &row);
void appendCol(cv::Mat &dstMat, cv::Mat &col);

// math things
void nchoosek(int n, int k, std::deque<std::vector<unsigned int>> &outputCombinations);
double erf(double x);
double erfc(double x);
cv::Mat histogram(cv::Mat singleChannelImage, int numBin);
bool IsLineSegmentIntersect(PSN_Line &line1, PSN_Line &line2);

// display related
std::vector<cv::Scalar> GenerateColors(unsigned int numColor);
cv::Scalar hsv2rgb(double h, double s, double v);
cv::Scalar getColorByID(std::vector<cv::Scalar> &vecColors, unsigned int nID);
void DrawBoxWithID(cv::Mat &imageFrame, PSN_Rect curRect, unsigned int nID, std::vector<cv::Scalar> &vecColors);
void DrawBoxWithLargeID(cv::Mat &imageFrame, PSN_Rect curRect, unsigned int nID, std::vector<cv::Scalar> &vecColors, bool bDashed = false);
void Draw3DBoxWithID(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, std::vector<cv::Scalar> &vecColors);
void DrawTriangleWithID(cv::Mat &imageFrame, PSN_Point2D &point, unsigned int nID, std::vector<cv::Scalar> &vecColors);
void DrawLine(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, std::vector<cv::Scalar> &vecColors, int lineThickness = 2);

// database related coordinate transformation
PSN_Point2D GetLocationOnTopView_PETS2009(PSN_Point3D &curPoint, bool bZoom = false);

// file interface related
void printLog(const char *filename, std::string strLog);
std::string MakeTrackIDList(PSN_TrackSet *tracks);
}

class CPSNWhere_Manager
{
	//////////////////////////////////////////////////////////////////////////
	// INSTANCE VARIABLES
	//////////////////////////////////////////////////////////////////////////
private:

	//////////////////////////////////////////////////////////////////////////
	// INSTANCE METHODS
	//////////////////////////////////////////////////////////////////////////
public:
	CPSNWhere_Manager(void);
	~CPSNWhere_Manager(void);
	
	
	//////////////////////////////////////////////////////////////////////////
	// STATIC METHODS
	//////////////////////////////////////////////////////////////////////////
public:

	//----------------------------------------------------------------
	// Helpers
	//----------------------------------------------------------------
	static cv::Mat MakeMatTile(std::vector<cv::Mat> *imageArray, unsigned int numRows, unsigned int numCols);
	static std::vector<stDetection> ReadDetectionResultWithTxt(std::string strDatasetPath, unsigned int camIdx, unsigned int frameIdx);
	static std::vector<stTrack2DResult> Read2DTrackResultWithTxt(std::string strDatasetPath, unsigned int frameIdx);
	static double Triangulation(PSN_Line &line1, PSN_Line &line2, PSN_Point3D &midPoint3D);
};


