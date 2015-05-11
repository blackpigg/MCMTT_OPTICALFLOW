/******************************************************************************
 * Name : PSNWhere_Types
 * Date : 2015.05.09
 * Author : HAANJU.YOO
 * Version : 0.9
 * Description :
 *	Define common types for system
 ******************************************************************************/
#pragma once

#include <list>
#include "PSNWhere_Defines.h"
#include "PSNWhere_SGSmooth.h"
#include "calibration\cameraModel.h"
#include "cv.h"

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

typedef std::vector<stTrackInfo> TrackInfoVector;
//typedef CArray<stTrackInfo, stTrackInfo&> TrackInfoArray;			// MFC, 한 시점에서 존재하는 모든 물체에 대한 트래킹 정보. 길이 = 물체 개수

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

class CTrackletSet
{
private:
	stTracklet2D *tracklets[NUM_CAM];
	unsigned int numTracklets;

public:
	CTrackletSet(void);

	// operator
	CTrackletSet& operator=(const CTrackletSet &a);
	bool operator==(const CTrackletSet &a);

	// methods
	void set(unsigned int camIdx, stTracklet2D *tracklet);
	stTracklet2D* get(unsigned int camIdx);
	void print(void);
	bool checkCoupling(CTrackletSet &a);
	unsigned int size(void);
};

struct stReconstruction
{
	bool bIsMeasurement;
	CTrackletSet tracklet2Ds;
	std::vector<PSN_Point3D> rawPoints;
	PSN_Point3D point;
	PSN_Point3D smoothedPoint;
	PSN_Point3D velocity;
	double detectionProbabilityRatio;
	double costReconstruction;
	double costSmoothedPoint;
	double costLink;
};

class CPointSmoother
{
public:
	CPointSmoother(void);
	~CPointSmoother(void);
	int Insert(PSN_Point3D &point);
	int Insert(std::vector<PSN_Point3D> &points);
	int ReplaceBack(PSN_Point3D &point);
	void PopBack(void);
	PSN_Point3D GetResult(int pos);
	std::vector<PSN_Point3D> GetResults(int startPos, int endPos = -1);
	size_t size(void) const { return size_; }
	PSN_Point3D back(void) const { return *back_; }

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
	void Initialize(unsigned int id, Track3D *parentTrack, unsigned int timeGeneration, CTrackletSet &trackletSet);
	int InsertReconstruction(stReconstruction &reconstruction);
	int InsertReconstructions(std::vector<stReconstruction> &reconstructions);
	int ReplaceBackReconstruction(stReconstruction &reconstruction);
	void RemoveFromTree();
	static std::deque<Track3D*> GatherValidChildrenTracks(Track3D* validParentTrack, std::deque<Track3D*> &targetChildrenTracks);
	double GetCost(void);

	unsigned int id;
	CTrackletSet curTracklet2Ds;
	std::deque<unsigned int> tracklet2DIDs[NUM_CAM];
	PSN_Point3D trackletLastLocation3D[NUM_CAM];
	bool bActive;	// for update and branching
	bool bValid;	// for deletion
	bool bNewTrack; // for HO-MHT
	bool bWasBestSolution;
	bool bCurrentBestSolution;	
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
	// trajectory related	
	CPointSmoother pointSmoother;
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
	// appearance
	cv::Mat lastRGBFeature[NUM_CAM];
	unsigned int timeTrackletEnded[NUM_CAM];
};

typedef std::deque<Track3D*> PSN_TrackSet;

struct stTracklet2DInfo
{
	stTracklet2D *tracklet2D;
	std::deque<Track3D*> queueRelatedTracks;
};

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
	unsigned int numMeasurements;
	std::deque<stTracklet2DInfo> tracklet2Ds[NUM_CAM];
public:
	TrackTree();
	~TrackTree();
	void Initialize(unsigned int id, Track3D *seedTrack, unsigned int timeGeneration, std::list<TrackTree> &treeList);
	void InsertTrack(Track3D *track);
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
	static bool CheckConnectivityOfTrees(TrackTree *tree1, TrackTree *tree2);
	

private:
};