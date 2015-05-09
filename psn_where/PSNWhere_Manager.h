/******************************************************************************
 * Name : PSNWhere_Manager
 * Date : 2015.05.09
 * Author : HAANJU.YOO
 * Version : 0.9
 * Description :
 *	Define miscellernous functions
 ******************************************************************************/

#pragma once

#include "stdafx.h"
#include <queue>
#include <time.h>
#include <list>

#include "PSNWhere_Defines.h"
#include "PSNWhere_Types.h"
#include "cv.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "calibration\cameraModel.h"
#include "PSNWhere_SGSmooth.h"


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
void printLog(const char *filename, const char *strLog);
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
	static void printLog(const char *filename, const char *strLog);
	static double Triangulation(PSN_Line &line1, PSN_Line &line2, PSN_Point3D &midPoint3D);
};


