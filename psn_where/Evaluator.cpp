#include "Evaluator.h"
#include <numeric>

//#define CROP_ZONE_X_MIN (-14069.6)
//#define CROP_ZONE_X_MAX (4981.3)
//#define CROP_ZONE_Y_MIN (-14274.0)
//#define CROP_ZONE_Y_MAX (1733.5)
//#define CROP_ZONE_MARGIN (1000.0)
#define MAXIMUM_POINT_ERROR (CROP_ZONE_MARGIN)
#define BOUNDARY_PROCESSING_ (true)

CEvaluator::CEvaluator(void)
	: bInit(false)
{
}


CEvaluator::~CEvaluator(void)
{
}

void CEvaluator::Initialize(std::string strFilepath)
{
	if (this->bInit) { return; }
	this->m_nSavedResult = 0;

	// measures	
	this->m_stResult.fMOTA = 0.0;
	this->m_stResult.fMOTP = 0.0;
	this->m_stResult.fMOTAL = 0.0;
	this->m_stResult.fRecall = 0.0;
	this->m_stResult.fPrecision = 0.0;
	this->m_stResult.fMissTargetPerGroundTruth = 0.0;
	this->m_stResult.fFalseAlarmPerGroundTruth = 0.0;
	this->m_stResult.fFalseAlarmPerFrame = 0.0;

	this->m_stResult.nMissed = 0;
	this->m_stResult.nFalsePositives = 0;
	this->m_stResult.nIDSwitch = 0;
	this->m_stResult.nMostTracked = 0;
	this->m_stResult.nPartilalyTracked = 0;
	this->m_stResult.nMostLost = 0;	
	this->m_stResult.nFragments = 0;

	// read ground truth
	FILE *fp;	
	char strPath[128];
	sprintf_s(strPath, "%s/groundTruth/cropped.txt", strFilepath.c_str());
	try
	{
		fopen_s(&fp, strPath, "r");
		float tempFloat;

		fscanf_s(fp, "numObj=%d,numTime=%d\n", &this->m_nNumObj, &this->m_nNumTime);
		this->matXgt = cv::Mat(this->m_nNumTime, this->m_nNumObj, CV_64FC1);
		this->matYgt = cv::Mat(this->m_nNumTime, this->m_nNumObj, CV_64FC1);
		this->m_queueSavedResult.resize(this->m_nNumTime);

		// read X
		fscanf_s(fp, "X={\n");
		for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
		{
			for (int objIdx = 0; objIdx < this->m_nNumObj; objIdx++)
			{
				fscanf_s(fp, "%f,", &tempFloat);
				this->matXgt.at<double>(timeIdx, objIdx) = (double)tempFloat;
			}
			fscanf_s(fp, "\n");
		}

		// read Y
		fscanf_s(fp, "}\nY={\n");
		for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
		{
			for (int objIdx = 0; objIdx < this->m_nNumObj; objIdx++)
			{
				fscanf_s(fp, "%f,", &tempFloat);
				this->matYgt.at<double>(timeIdx, objIdx) = (double)tempFloat;
			}
			fscanf_s(fp, "\n");
		}

		fclose(fp);
	}
	catch (DWORD dwError)
	{
		printf("[ERROR] cannot open data. error code %d\n", dwError);
	}

	// cropzone setting
	this->m_rectCropZone.x = CROP_ZONE_X_MIN;
	this->m_rectCropZone.y = CROP_ZONE_Y_MIN;
	this->m_rectCropZone.w = CROP_ZONE_X_MAX - CROP_ZONE_X_MIN;
	this->m_rectCropZone.h = CROP_ZONE_Y_MAX - CROP_ZONE_Y_MIN;

	this->m_rectCropZoneMargin = this->m_rectCropZone;
	this->m_rectCropZoneMargin.x -= CROP_ZONE_MARGIN;
	this->m_rectCropZoneMargin.x -= CROP_ZONE_MARGIN;
	this->m_rectCropZoneMargin.w += 2 * CROP_ZONE_MARGIN;
	this->m_rectCropZoneMargin.h += 2 * CROP_ZONE_MARGIN;

	this->bInit = true;
}

void CEvaluator::Finalize(void)
{
	if (!this->bInit) {	return; }

	this->matXgt.release();
	this->matYgt.release();
	this->matY.release();
	this->matX.release();
	this->m_queueID.clear();
	this->m_queueSavedResult.clear();

	this->bInit = false;
}

void CEvaluator::SetResult(PSN_TrackSet &trackSet, unsigned int timeIdx)
{
	if (timeIdx > this->m_queueSavedResult.size()) { return; }

	this->m_queueSavedResult[timeIdx].clear();
	for (PSN_TrackSet::iterator trackIter = trackSet.begin();
		trackIter != trackSet.end();
		trackIter++)
	{
		int reconIdx = (int)timeIdx - (*trackIter)->timeStart;
		if (0 > reconIdx || (int)(*trackIter)->reconstructions.size() <= reconIdx) { continue;	}

		PSN_Point3D curPoint = (*trackIter)->reconstructions[timeIdx - (*trackIter)->timeStart].smoothedPoint;
		if ((BOUNDARY_PROCESSING_ && !this->m_rectCropZoneMargin.contain(PSN_Point2D(curPoint.x, curPoint.y)))
			|| (!BOUNDARY_PROCESSING_ && !this->m_rectCropZone.contain(PSN_Point2D(curPoint.x, curPoint.y)))) 
		{ continue; }

		// index management
		int indexPos = 0;
		std::deque<unsigned int>::iterator findIter = std::find(this->m_queueID.begin(), this->m_queueID.end(), (*trackIter)->tree->id);
		if (this->m_queueID.end() == findIter)
		{
			this->m_queueID.push_back((*trackIter)->tree->id);
			indexPos = (int)this->m_queueID.size() - 1;
		}
		else
		{
			indexPos = (int)(findIter - this->m_queueID.begin());
		}		
		this->m_queueSavedResult[timeIdx].push_back(std::make_pair(indexPos, curPoint));
	}
	this->m_nSavedResult++;
}

void CEvaluator::LoadResultFromText(std::string strFilepath)
{
	FILE *fp;	
	char strPath[128];
	sprintf_s(strPath, "%s", strFilepath.c_str());
	try
	{
		fopen_s(&fp, strPath, "r");

		float readingFloat;
		int numObject = 0;
		fscanf_s(fp, "MatX:(%d,%d)\n", &this->m_nNumTime, &numObject);
		this->m_queueSavedResult.resize(this->m_nNumTime);
		this->m_queueID.clear();
		this->m_nSavedResult = this->m_nNumTime;
		for (int objIdx = 0; objIdx < numObject; objIdx++) { this->m_queueID.push_back(objIdx); }

		// read X
		for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
		{
			for (int objIdx = 0; objIdx < numObject; objIdx++)
			{
				fscanf_s(fp, "%f,", &readingFloat);
				if (0.0 == readingFloat) { continue; }
				this->m_queueSavedResult[timeIdx].push_back(std::make_pair(objIdx, PSN_Point3D((double)readingFloat, 0.0, 0.0)));
			}
			fscanf_s(fp, "\n");			
		}

		// read Y
		fscanf_s(fp, "MatY:(%d,%d)\n", &this->m_nNumTime, &numObject);
		for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
		{
			for (int objIdx = 0; objIdx < numObject; objIdx++)
			{
				fscanf_s(fp, "%f,", &readingFloat);
				if (0.0 == readingFloat) { continue; }

				bool bFound = false;
				for (int findIdx = 0; findIdx < (int)this->m_queueSavedResult[timeIdx].size(); findIdx++)
				{
					if (objIdx == this->m_queueSavedResult[timeIdx][findIdx].first)
					{
						bFound = true;
						this->m_queueSavedResult[timeIdx][findIdx].second.y = (double)readingFloat;
						break;
					}
				}

				if (!bFound)
				{
					this->m_queueSavedResult[timeIdx].push_back(std::make_pair(objIdx, PSN_Point3D(0.0, (double)readingFloat, 0.0)));
				}
			}
			fscanf_s(fp, "\n");
		}

		fclose(fp);

		// crop
		for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
		{
			pointInfoSet croppedPointInfoSet;
			for (pointInfoSet::iterator pointInfoIter = this->m_queueSavedResult[timeIdx].begin();
			pointInfoIter != this->m_queueSavedResult[timeIdx].end();
			pointInfoIter++)
			{
				PSN_Point2D curPoint((*pointInfoIter).second.x, (*pointInfoIter).second.y);			
				if ((BOUNDARY_PROCESSING_ && !this->m_rectCropZoneMargin.contain(PSN_Point2D(curPoint.x, curPoint.y)))
					|| (!BOUNDARY_PROCESSING_ && !this->m_rectCropZone.contain(PSN_Point2D(curPoint.x, curPoint.y)))) 
				{ continue; }
				croppedPointInfoSet.push_back(*pointInfoIter);
			}
			this->m_queueSavedResult[timeIdx] = croppedPointInfoSet;
		}

	}
	catch (DWORD dwError)
	{
		printf("[ERROR] cannot open data. error code %d\n", dwError);
	}
}

void CEvaluator::Evaluate(void)
{
	//---------------------------------------------------------
	// CROP
	//---------------------------------------------------------	
	// zone setting
	PSN_Rect rectInnerCropZone = this->m_rectCropZone;
	rectInnerCropZone.x += CROP_ZONE_MARGIN;
	rectInnerCropZone.y += CROP_ZONE_MARGIN;
	rectInnerCropZone.w -= 2.0 * CROP_ZONE_MARGIN;
	rectInnerCropZone.h -= 2.0 * CROP_ZONE_MARGIN;

	// cropping and saving result with boundary points
	cv::Mat matX_b = cv::Mat::zeros(this->m_nNumTime, (int)this->m_queueID.size(), CV_64FC1);
	cv::Mat matY_b = cv::Mat::zeros(this->m_nNumTime, (int)this->m_queueID.size(), CV_64FC1);
	std::vector<pointInfoSet> queueCroppedResult(this->m_nNumTime);
	std::vector<pointInfoSet> queueInnerCroppedResult(this->m_nNumTime);
	for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
	{
		for (pointInfoSet::iterator pointInfoIter = this->m_queueSavedResult[timeIdx].begin();
			pointInfoIter != this->m_queueSavedResult[timeIdx].end();
			pointInfoIter++)
		{
			PSN_Point2D curPoint((*pointInfoIter).second.x, (*pointInfoIter).second.y);
			matX_b.at<double>(timeIdx, (*pointInfoIter).first) = (*pointInfoIter).second.x;
			matY_b.at<double>(timeIdx, (*pointInfoIter).first) = (*pointInfoIter).second.y;

			if (!this->m_rectCropZone.contain(curPoint)) { continue; }
			queueCroppedResult[timeIdx].push_back(*pointInfoIter);

			if (!rectInnerCropZone.contain(curPoint)) { continue; }
			queueInnerCroppedResult[timeIdx].push_back(*pointInfoIter);
		}
	}
	
	//---------------------------------------------------------
	// GENERATING RESULT MATRIX
	//---------------------------------------------------------	
	// generate result matrix
	if (!this->matX.empty()) { this->matX.release(); }
	if (!this->matY.empty()) { this->matY.release(); }
	this->matX = cv::Mat::zeros(this->m_nNumTime, (int)this->m_queueID.size(), CV_64FC1);
	this->matY = cv::Mat::zeros(this->m_nNumTime, (int)this->m_queueID.size(), CV_64FC1);
	cv::Mat matX_ic = cv::Mat::zeros(this->m_nNumTime, (int)this->m_queueID.size(), CV_64FC1);
	cv::Mat matY_ic = cv::Mat::zeros(this->m_nNumTime, (int)this->m_queueID.size(), CV_64FC1);
	for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
	{
		for (pointInfoSet::iterator pointInfoIter = queueCroppedResult[timeIdx].begin();
			pointInfoIter != queueCroppedResult[timeIdx].end();
			pointInfoIter++)
		{			
			this->matX.at<double>(timeIdx, (*pointInfoIter).first) = (*pointInfoIter).second.x;
			this->matY.at<double>(timeIdx, (*pointInfoIter).first) = (*pointInfoIter).second.y;
		}
		for (pointInfoSet::iterator pointInfoIter = queueInnerCroppedResult[timeIdx].begin();
			pointInfoIter != queueInnerCroppedResult[timeIdx].end();
			pointInfoIter++)
		{			
			matX_ic.at<double>(timeIdx, (*pointInfoIter).first) = (*pointInfoIter).second.x;
			matY_ic.at<double>(timeIdx, (*pointInfoIter).first) = (*pointInfoIter).second.y;
		}
	}

	//// processing for boundary
	//if (BOUNDARY_PROCESSING_)
	//{		
	//	PSN_Point2D curPoint;
	//	PSN_Point2D prevPoint, nextPoint;
	//	bool bPrevIn, bNextIn;
	//	for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
	//	{
	//		for (int objIdx = 0; objIdx < (int)this->m_queueID.size(); objIdx++)
	//		{
	//			curPoint.x = this->matX.at<double>(timeIdx, objIdx);
	//			curPoint.y = this->matY.at<double>(timeIdx, objIdx);
	//		
	//			if (this->m_rectCropZone.contain(curPoint)) { continue; }

	//			// if point in margin region
	//			bPrevIn = false;
	//			bNextIn = false;
	//			if (timeIdx > 0)
	//			{
	//				prevPoint.x = this->matX.at<double>(timeIdx-1, objIdx);
	//				prevPoint.y = this->matY.at<double>(timeIdx-1, objIdx);
	//				if(this->m_rectCropZone.contain(prevPoint))
	//				{
	//					bPrevIn = true;
	//				}
	//			}

	//			if (timeIdx < this->m_nNumTime - 1)
	//			{
	//				nextPoint.x = this->matX.at<double>(timeIdx+1, objIdx);
	//				nextPoint.y = this->matY.at<double>(timeIdx+1, objIdx);
	//				if (this->m_rectCropZone.contain(nextPoint))
	//				{
	//					bNextIn = true;
	//				}
	//			}

	//			if (bPrevIn && bNextIn)
	//			{
	//				// 1) in->out->in
	//			}
	//			else if (bPrevIn)
	//			{
	//				// 2) in->out
	//			}
	//			else if (bNextIn)
	//			{
	//				// 3) out->in
	//			}				
	//		}
	//	}
	//}

	//---------------------------------------------------------
	// CROP GROUND TRUTH BY TIME INDEX
	//---------------------------------------------------------	
	int newNumTime = this->m_nSavedResult;
	cv::Mat newXgtCandidate = this->matXgt(cv::Rect(0, 0, this->matXgt.cols, newNumTime)).clone().t();
	cv::Mat newYgtCandidate = this->matYgt(cv::Rect(0, 0, this->matYgt.cols, newNumTime)).clone().t();
	cv::Mat newXgt, newYgt;
	int newNumObject = 0;
	for (int objectIdx = 0; objectIdx < this->m_nNumObj; objectIdx++)
	{
		if (0 == cv::countNonZero(newXgtCandidate.row(objectIdx)) || 0 == cv::countNonZero(newYgtCandidate.row(objectIdx)))
		{
			continue;
		}
		newXgt.push_back(newXgtCandidate.row(objectIdx));
		newYgt.push_back(newYgtCandidate.row(objectIdx));
		newNumObject++;
	}
	this->matXgt = newXgt.t();
	this->matYgt = newYgt.t();
	this->m_nNumObj = newNumObject;
	this->m_nNumTime = newNumTime;
	
	//---------------------------------------------------------
	// EVALUATING (porting from CLEAR_MOT.m)
	//---------------------------------------------------------
	stEvaluationResult curResult;
	int Fgt = this->m_nNumTime;
	int Ngt = this->m_nNumObj;
	//int F = (int)this->m_queueSavedResult.size();
	int F = this->m_nSavedResult;
	int N = (int)this->m_queueID.size();

	if(0 == N)
	{
		curResult.fMOTA = 0.0;
		curResult.fMOTP = 0.0;
		curResult.fMOTAL = 0.0;
		curResult.fRecall = 0.0;
		curResult.fPrecision = 0.0;
		curResult.fMissTargetPerGroundTruth = 1.0;
		curResult.fFalseAlarmPerGroundTruth = 0.0;
		curResult.fFalseAlarmPerFrame = 0.0;
		curResult.nMissed = cv::countNonZero(this->matXgt);
		curResult.nMostTracked = 0;
		curResult.nPartilalyTracked = 0;
		curResult.nMostLost = this->m_nNumObj;
		curResult.nIDSwitch = 0;
		curResult.nFragments = 0;

		this->m_stResult = curResult;

		return;
	}

	cv::Mat M = cv::Mat(F, Ngt, CV_32SC1, -1);	// index of C starts from zero	
	std::vector<int> mme(F, 0);		// ID switches
	std::vector<int> c(F, 0);		// matches found
	std::vector<int> fp(F, 0);		// false positives
	std::vector<int> m(F, 0);		// misses = false negatives
	std::vector<int> g(F, 0);
	cv::Mat d = cv::Mat::zeros(F, Ngt, CV_64FC1);	// all distances

	PSN_Point2D curGtPoint;
	PSN_Point2D curResPoint;
	for (int t = 0; t < F; t++)
	{
		g[t] = cv::countNonZero(this->matXgt.row(t));

		// inherent matching
		if (t > 0)
		{
			for(int mapIdx = 0; mapIdx < (int)M.cols; mapIdx++)
			{
				if (-1 == M.at<int>(t-1, mapIdx)) { continue; }

				// get ground truth point
				curGtPoint.x = this->matXgt.at<double>(t, mapIdx);
				curGtPoint.y = this->matYgt.at<double>(t, mapIdx);
				if (0.0 == curGtPoint.x || 0.0 == curGtPoint.y) { continue; }

				// get result point
				if (0.0 != this->matX.at<double>(t, M.at<int>(t-1, mapIdx)))
				{
					// it is sufficient to check only x coordinate!!
					curResPoint.x = this->matX.at<double>(t, M.at<int>(t-1, mapIdx));
					curResPoint.y = this->matY.at<double>(t, M.at<int>(t-1, mapIdx));
				}
				else
				{
					curResPoint.x = matX_b.at<double>(t, M.at<int>(t-1, mapIdx));
					curResPoint.y = matY_b.at<double>(t, M.at<int>(t-1, mapIdx));
				}

				// check matching condition
				if (0.0 == curResPoint.x || 0.0 == curResPoint.y) { continue; }
				if ((curGtPoint - curResPoint).norm_L2() > MAXIMUM_POINT_ERROR) { continue; }

				// matched!!
				M.at<int>(t, mapIdx) = M.at<int>(t-1, mapIdx);

				// original
				//curGtPoint.x = this->matXgt.at<double>(t, mapIdx);
				//curGtPoint.y = this->matYgt.at<double>(t, mapIdx);
				//curResPoint.x = this->matX.at<double>(t, M.at<int>(t-1, mapIdx));
				//curResPoint.y = this->matY.at<double>(t, M.at<int>(t-1, mapIdx));
				//if(0.0 != curGtPoint.x && 0.0 != curGtPoint.y && 0.0 != curResPoint.x && 0.0 != curResPoint.y
				//	&& (curGtPoint - curResPoint).norm_L2() <= MAXIMUM_POINT_ERROR)
				//{
				//	M.at<int>(t, mapIdx) = M.at<int>(t-1, mapIdx);
				//}
			}
		}

		// matching
		cv::Mat allDist = cv::Mat(Ngt, N, CV_64FC1, FLT_MAX);
		std::deque<int> GTsNotMapped;
		std::deque<int> EsNotMapped;
		double minDist = 0.0, curDist = 0.0;
		int minIdxGT = 0, minIdxE = 0;
		PSN_Point2D GT, E;
		do
		{
			GTsNotMapped.clear();
			EsNotMapped.clear();
			for (int colIdx = 0; colIdx < Ngt; colIdx++)
			{
				if (-1 == M.at<int>(t, colIdx) && 0.0 != this->matXgt.at<double>(t, colIdx))
				{
					GTsNotMapped.push_back(colIdx);
				}
				//if(-1 == M.at<int>(t, colIdx) && 0.0 != this->matX.at<double>(t, colIdx))
				//{
				//	EsNotMapped.push_back(colIdx);
				//}
			}
			for (int colIdx = 0; colIdx < (int)this->matX.cols; colIdx++)
			{
				if (0.0 != this->matX.at<double>(t, colIdx))
				{
					bool bFound = false;
					for (int mColIdx = 0; mColIdx < Ngt; mColIdx++)
					{
						if (colIdx == M.at<int>(t, mColIdx))
						{
							bFound = true;
							break;
						}
					}
					if (!bFound)
					{
						EsNotMapped.push_back(colIdx);
					}
				}
			}

			minDist = FLT_MAX;
			minIdxGT = 0;
			minIdxE = 0;
			for (int o = 0; o < (int)GTsNotMapped.size(); o++)
			{
				GT.x = this->matXgt.at<double>(t, GTsNotMapped[o]);
				GT.y = this->matYgt.at<double>(t, GTsNotMapped[o]);
				for (int e = 0; e < (int)EsNotMapped.size(); e++)
				{
					E.x = this->matX.at<double>(t, EsNotMapped[e]);
					E.y = this->matY.at<double>(t, EsNotMapped[e]);
					curDist = (GT - E).norm_L2();
					if (curDist < minDist)
					{
						minDist = curDist;
						minIdxGT = GTsNotMapped[o];
						minIdxE = EsNotMapped[e];
					}
				}
			}

			if (minDist > CROP_ZONE_MARGIN) { break; }
			M.at<int>(t, minIdxGT) = minIdxE;
		} while (minDist < CROP_ZONE_MARGIN && GTsNotMapped.size() > 0 && EsNotMapped.size() > 0);

		// mismatch errors
		c[t] = 0;		
		for (int ct = 0; ct < Ngt; ct++)
		{
			if (-1 == M.at<int>(t, ct)) { continue; }
			c[t]++;

			if (t > 0)
			{
				int lastNotEmpty = -1;
				for (int tIdx = 0; tIdx < t; tIdx++)
				{
					if (-1 != M.at<int>(tIdx, ct)) { lastNotEmpty = tIdx; }
				}

				if (0.0 != this->matXgt.at<double>(t-1, ct) && -1 != lastNotEmpty && M.at<int>(t, ct) != M.at<int>(lastNotEmpty, ct))
				{ mme[t]++; }
			}

			// distance
			int eid = M.at<int>(t, ct);
			curGtPoint.x = this->matXgt.at<double>(t, ct);
			curGtPoint.y = this->matYgt.at<double>(t, ct);
			if (0 != this->matX.at<double>(t, eid))
			{
				curResPoint.x = this->matX.at<double>(t, eid);
				curResPoint.y = this->matY.at<double>(t, eid);
			}
			else
			{
				curResPoint.x = matX_b.at<double>(t, eid);
				curResPoint.y = matY_b.at<double>(t, eid);
			}
			d.at<double>(t, ct) = (curGtPoint - curResPoint).norm_L2();
		}

		// false positive		
		fp[t] = 0;
		for (int objIdx = 0; objIdx < (int)this->matX.cols; objIdx++)
		{
			if (0.0 == this->matX.at<double>(t, objIdx)) { continue; }

			// matching check
			std::deque<int>::iterator findIter = std::find(EsNotMapped.begin(), EsNotMapped.end(), objIdx);
			if (EsNotMapped.end() == findIter) { continue; }
			fp[t]++;		

			// NEOHANJU(2015.07.15): handling inner padding with 'NumBoundaryPadPoints'
			if (0.0 != matX_ic.at<double>(t, objIdx)) { continue; }			

			// connectivity check (check whether this point is hurry or dragging point)
			if (0 == t && t < F-1)
			{
				if (0.0 == this->matX.at<double>(t+1, objIdx)) { continue; }
			}
			else if (t < F-1)
			{
				if (0.0 == this->matX.at<double>(t-1, objIdx) && 0.0 == this->matX.at<double>(t+1, objIdx)) { continue; }
			}
			else if (0.0 == this->matX.at<double>(t-1, objIdx))
			{
				continue;
			}
			fp[t]--;
		}
		//fp[t] -= c[t] + NumBoundaryPadPoints; // original: fp[t] -= c[t];

		// miss (false negative)
		m[t] = g[t] - c[t];
	}

	// measurement calculation
	double sumC = (double)std::accumulate(c.begin(), c.end(), 0);
	double sumG = (double)std::accumulate(g.begin(), g.end(), 0);
	double sumM = (double)std::accumulate(m.begin(), m.end(), 0);
	double sumFP = (double)std::accumulate(fp.begin(), fp.end(), 0);
	double sumMME = (double)std::accumulate(mme.begin(), mme.end(), 0);

	curResult.nMissed = (int)sumM;
	curResult.nFalsePositives = (int)sumFP;
	curResult.nIDSwitch = (int)sumMME;

	curResult.fMOTP = 1.0 - (double)cv::sum(d)[0] / (sumC * (double)CROP_ZONE_MARGIN);
	curResult.fMOTA = 1.0 - (sumM + sumFP + sumMME) / sumG;
	curResult.fMOTAL = 1.0 - ((sumM + sumFP + std::log10(sumMME + 1))) / sumG;
	curResult.fMissTargetPerGroundTruth = sumM / sumG;
	curResult.fFalseAlarmPerGroundTruth = sumFP / sumG;
	curResult.fRecall = sumC / sumG;
	curResult.fPrecision = sumC / (sumFP + sumC);
	curResult.fFalseAlarmPerFrame = sumFP / Fgt;

	// MT PT ML
	std::vector<int> MTstatsa(Ngt, 0);
	curResult.nMostTracked = 0;
	curResult.nPartilalyTracked = 0;
	curResult.nMostLost = 0;
	for (int i = 0; i < Ngt; i++)
	{
		double getLength = 0;
		double trackedLength = 0;
		int lastIndex;
		for (int tIdx = 0; tIdx < Fgt; tIdx++)
		{
			if (0 != this->matXgt.at<double>(tIdx, i))
			{
				getLength++;
				lastIndex = tIdx;
				if (0 <= M.at<int>(tIdx, i)) { trackedLength++; }
			}
		}

		if (trackedLength / getLength < 0.2)
		{
			MTstatsa[i] = 3;
			curResult.nMostLost++;
		}
		else if (F >= lastIndex && trackedLength / getLength <= 0.8)
		{
			MTstatsa[i] = 2;
			curResult.nPartilalyTracked++;
		}
		else if (trackedLength / getLength >= 0.8)
		{
			MTstatsa[i] = 1;
			curResult.nMostTracked++;
		}
	}

	// fragments
	std::vector<int> fr(Ngt, 0);
	curResult.nFragments = 0;
	for (int i = 0; i < Ngt; i++)
	{
		int startIdx = 0;
		int endIdx = 0;
		int numSwtich = 0;
		bool bTracked = false;
		bool bStart = false;
		for (int tIdx = 0; tIdx < Fgt; tIdx++)
		{
			if (0 <= M.at<int>(tIdx, i))
			{
				if (bStart) { endIdx = tIdx; }
				else
				{
					bStart = true;					
					startIdx = tIdx;
				}
				bTracked = true;
			}
			else
			{
				if (bTracked) {	numSwtich++; }
				bTracked = false;
			}
		}
		if (Fgt - 1 > endIdx) {	numSwtich--; }
		curResult.nFragments += numSwtich;
	}	

	this->m_stResult = curResult;
}

//stEvaluationResult CEvaluator::EvaluateWithCrop(double cropMargin)
//{
//	//---------------------------------------------------------
//	// CROP
//	//---------------------------------------------------------	
//	// zone setting
//	PSN_Rect rectCropZone = m_rectCropZone;
//	rectCropZone.x -= cropMargin;
//	rectCropZone.y -= cropMargin;
//	rectCropZone.w += 2.0 * cropMargin;
//	rectCropZone.h += 2.0 * cropMargin;
//
//	// cropping
//	std::vector<pointInfoSet> queueCroppedResult(this->m_nNumTime);
//	for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
//	{
//		for (pointInfoSet::iterator pointInfoIter = this->m_queueSavedResult[timeIdx].begin();
//			pointInfoIter != this->m_queueSavedResult[timeIdx].end();
//			pointInfoIter++)
//		{
//			PSN_Point2D curPoint((*pointInfoIter).second.x, (*pointInfoIter).second.y);
//			if (!rectCropZone.contain(curPoint)) { continue; }
//			queueCroppedResult[timeIdx].push_back(*pointInfoIter);
//		}
//	}
//	
//	//---------------------------------------------------------
//	// GENERATING RESULT MATRIX
//	//---------------------------------------------------------	
//	// generate result matrix
//	if (!this->matX.empty()) { this->matX.release(); }
//	if (!this->matY.empty()) { this->matY.release(); }
//	this->matX = cv::Mat::zeros(this->m_nNumTime, (int)this->m_queueID.size(), CV_64FC1);
//	this->matY = cv::Mat::zeros(this->m_nNumTime, (int)this->m_queueID.size(), CV_64FC1);
//	for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
//	{
//		for (pointInfoSet::iterator pointInfoIter = queueCroppedResult[timeIdx].begin();
//			pointInfoIter != queueCroppedResult[timeIdx].end();
//			pointInfoIter++)
//		{			
//			this->matX.at<double>(timeIdx, (*pointInfoIter).first) = (*pointInfoIter).second.x;
//			this->matY.at<double>(timeIdx, (*pointInfoIter).first) = (*pointInfoIter).second.y;
//		}
//	}
//
//	// processing for boundary
//	if (BOUNDARY_PROCESSING_)
//	{		
//		PSN_Point2D curPoint;
//		PSN_Point2D prevPoint, nextPoint;
//		bool bPrevIn, bNextIn;
//		for (int timeIdx = 0; timeIdx < this->m_nNumTime; timeIdx++)
//		{
//			for (int objIdx = 0; objIdx < (int)this->m_queueID.size(); objIdx++)
//			{
//				curPoint.x = this->matX.at<double>(timeIdx, objIdx);
//				curPoint.y = this->matY.at<double>(timeIdx, objIdx);
//			
//				if (this->m_rectCropZone.contain(curPoint)) { continue; }
//
//				// if point in margin region
//				bPrevIn = false;
//				bNextIn = false;
//				if (timeIdx > 0)
//				{
//					prevPoint.x = this->matX.at<double>(timeIdx-1, objIdx);
//					prevPoint.y = this->matY.at<double>(timeIdx-1, objIdx);
//					if(this->m_rectCropZone.contain(prevPoint))
//					{
//						bPrevIn = true;
//					}
//				}
//
//				if (timeIdx < this->m_nNumTime - 1)
//				{
//					nextPoint.x = this->matX.at<double>(timeIdx+1, objIdx);
//					nextPoint.y = this->matY.at<double>(timeIdx+1, objIdx);
//					if (this->m_rectCropZone.contain(nextPoint))
//					{
//						bNextIn = true;
//					}
//				}
//
//				if (bPrevIn && bNextIn)
//				{
//					// 1) in->out->in
//				}
//				else if (bPrevIn)
//				{
//					// 2) in->out
//				}
//				else if (bNextIn)
//				{
//					// 3) out->in
//				}				
//			}
//		}
//	}
//
//	//---------------------------------------------------------
//	// CROP GROUND TRUTH BY TIME INDEX
//	//---------------------------------------------------------	
//	int newNumTime = this->m_nSavedResult;
//	cv::Mat newXgtCandidate = this->matXgt(cv::Rect(0, 0, this->matXgt.cols, newNumTime)).clone().t();
//	cv::Mat newYgtCandidate = this->matYgt(cv::Rect(0, 0, this->matYgt.cols, newNumTime)).clone().t();
//	cv::Mat newXgt, newYgt;
//	int newNumObject = 0;
//	for (int objectIdx = 0; objectIdx < this->m_nNumObj; objectIdx++)
//	{
//		if (0 == cv::countNonZero(newXgtCandidate.row(objectIdx)) || 0 == cv::countNonZero(newYgtCandidate.row(objectIdx)))
//		{
//			continue;
//		}
//		newXgt.push_back(newXgtCandidate.row(objectIdx));
//		newYgt.push_back(newYgtCandidate.row(objectIdx));
//		newNumObject++;
//	}
//	this->matXgt = newXgt.t();
//	this->matYgt = newYgt.t();
//	this->m_nNumObj = newNumObject;
//	this->m_nNumTime = newNumTime;
//	
//	//---------------------------------------------------------
//	// EVALUATING (porting from CLEAR_MOT.m)
//	//---------------------------------------------------------
//	stEvaluationResult curResult;
//	int Fgt = this->m_nNumTime;
//	int Ngt = this->m_nNumObj;
//	//int F = (int)this->m_queueSavedResult.size();
//	int F = this->m_nSavedResult;
//	int N = (int)this->m_queueID.size();
//
//	if(0 == N)
//	{
//		curResult.fMOTA = 0.0;
//		curResult.fMOTP = 0.0;
//		curResult.fMOTAL = 0.0;
//		curResult.fRecall = 0.0;
//		curResult.fPrecision = 0.0;
//		curResult.fMissTargetPerGroundTruth = 1.0;
//		curResult.fFalseAlarmPerGroundTruth = 0.0;
//		curResult.fFalseAlarmPerFrame = 0.0;
//		curResult.nMissed = cv::countNonZero(this->matXgt);
//		curResult.nMostTracked = 0;
//		curResult.nPartilalyTracked = 0;
//		curResult.nMostLost = this->m_nNumObj;
//		curResult.nIDSwitch = 0;
//		curResult.nFragments = 0;
//
//		return curResult;
//	}
//
//	cv::Mat M = cv::Mat(F, Ngt, CV_32SC1, -1);	// index of C starts from zero	
//	std::vector<int> mme(F, 0);		// ID switches
//	std::vector<int> c(F, 0);		// matches found
//	std::vector<int> fp(F, 0);		// false positives
//	std::vector<int> m(F, 0);		// misses = false negatives
//	std::vector<int> g(F, 0);
//	cv::Mat d = cv::Mat::zeros(F, Ngt, CV_64FC1);	// all distances
//
//	PSN_Point2D curGtPoint;
//	PSN_Point2D curResPoint;
//	for(int t = 0; t < F; t++)
//	{
//		g[t] = cv::countNonZero(this->matXgt.row(t));
//		// inherent matching
//		if(t > 0)
//		{
//			for(int mapIdx = 0; mapIdx < (int)M.cols; mapIdx++)
//			{
//				if(-1 == M.at<int>(t-1, mapIdx))
//				{ continue;	}
//
//				curGtPoint.x = this->matXgt.at<double>(t, mapIdx);
//				curGtPoint.y = this->matYgt.at<double>(t, mapIdx);
//				curResPoint.x = this->matX.at<double>(t, M.at<int>(t-1, mapIdx));
//				curResPoint.y = this->matY.at<double>(t, M.at<int>(t-1, mapIdx));
//
//				double norm = (curGtPoint - curResPoint).norm_L2();
//
//				if(0.0 != curGtPoint.x && 0.0 != curGtPoint.y && 0.0 != curResPoint.x && 0.0 != curResPoint.y
//					&& (curGtPoint - curResPoint).norm_L2() <= CROP_ZONE_MARGIN)
//				{
//					M.at<int>(t, mapIdx) = M.at<int>(t-1, mapIdx);
//				}
//			}
//		}
//
//		// matching
//		cv::Mat allDist = cv::Mat(Ngt, N, CV_64FC1, FLT_MAX);
//		std::deque<int> GTsNotMapped;
//		std::deque<int> EsNotMapped;
//		double minDist = 0.0, curDist = 0.0;
//		int minIdxGT = 0, minIdxE = 0;
//		PSN_Point2D GT, E;
//		do
//		{
//			GTsNotMapped.clear();
//			EsNotMapped.clear();
//			for (int colIdx = 0; colIdx < Ngt; colIdx++)
//			{
//				if (-1 == M.at<int>(t, colIdx) && 0.0 != this->matXgt.at<double>(t, colIdx))
//				{
//					GTsNotMapped.push_back(colIdx);
//				}
//				//if(-1 == M.at<int>(t, colIdx) && 0.0 != this->matX.at<double>(t, colIdx))
//				//{
//				//	EsNotMapped.push_back(colIdx);
//				//}
//			}
//			for (int colIdx = 0; colIdx < (int)this->matX.cols; colIdx++)
//			{
//				if (0.0 != this->matX.at<double>(t, colIdx))
//				{
//					bool bFound = false;
//					for (int mColIdx = 0; mColIdx < Ngt; mColIdx++)
//					{
//						if (colIdx == M.at<int>(t, mColIdx))
//						{
//							bFound = true;
//							break;
//						}
//					}
//					if (!bFound)
//					{
//						EsNotMapped.push_back(colIdx);
//					}
//				}
//			}
//
//			minDist = FLT_MAX;
//			minIdxGT = 0;
//			minIdxE = 0;
//			for (int o = 0; o < (int)GTsNotMapped.size(); o++)
//			{
//				GT.x = this->matXgt.at<double>(t, GTsNotMapped[o]);
//				GT.y = this->matYgt.at<double>(t, GTsNotMapped[o]);
//				for (int e = 0; e < (int)EsNotMapped.size(); e++)
//				{
//					E.x = this->matX.at<double>(t, EsNotMapped[e]);
//					E.y = this->matY.at<double>(t, EsNotMapped[e]);
//					curDist = (GT - E).norm_L2();
//					if (curDist < minDist)
//					{
//						minDist = curDist;
//						minIdxGT = GTsNotMapped[o];
//						minIdxE = EsNotMapped[e];
//					}
//				}
//			}
//
//			if (minDist > CROP_ZONE_MARGIN) { break; }
//			M.at<int>(t, minIdxGT) = minIdxE;
//		}
//
//		while (minDist < CROP_ZONE_MARGIN && GTsNotMapped.size() > 0 && EsNotMapped.size() > 0);
//
//		// mismatch errors
//		c[t] = 0;		
//		for (int ct = 0; ct < Ngt; ct++)
//		{
//			if (-1 == M.at<int>(t, ct)) { continue; }
//			c[t]++;
//
//			if (t > 0)
//			{
//				int lastNotEmpty = -1;
//				for (int tIdx = 0; tIdx < t; tIdx++)
//				{
//					if (-1 != M.at<int>(tIdx, ct)) { lastNotEmpty = tIdx; }
//				}
//
//				if (0.0 != this->matXgt.at<double>(t-1, ct) && -1 != lastNotEmpty && M.at<int>(t, ct) != M.at<int>(lastNotEmpty, ct))
//				{ mme[t]++; }
//			}
//
//			// distance
//			int eid = M.at<int>(t, ct);
//			curGtPoint.x = this->matXgt.at<double>(t, ct);
//			curGtPoint.y = this->matYgt.at<double>(t, ct);
//			curResPoint.x = this->matX.at<double>(t, eid);
//			curResPoint.y = this->matY.at<double>(t, eid);
//			d.at<double>(t, ct) = (curGtPoint - curResPoint).norm_L2();
//		}
//
//		fp[t] = 0;
//		for (int objIdx = 0; objIdx < (int)this->matX.cols; objIdx++)
//		{
//			if (0.0 != this->matX.at<double>(t, objIdx))
//			{ fp[t]++; }
//		}
//		fp[t] -= c[t];
//		m[t] = g[t] - c[t];
//	}
//
//	// measurement calculation
//	double sumC = (double)std::accumulate(c.begin(), c.end(), 0);
//	double sumG = (double)std::accumulate(g.begin(), g.end(), 0);
//	double sumM = (double)std::accumulate(m.begin(), m.end(), 0);
//	double sumFP = (double)std::accumulate(fp.begin(), fp.end(), 0);
//	double sumMME = (double)std::accumulate(mme.begin(), mme.end(), 0);
//
//	curResult.nMissed = (int)sumM;
//	curResult.nFalsePositives = (int)sumFP;
//	curResult.nIDSwitch = (int)sumMME;
//
//	curResult.fMOTP = 1.0 - (double)cv::sum(d)[0] / (sumC * (double)CROP_ZONE_MARGIN);
//	curResult.fMOTA = 1.0 - (sumM + sumFP + sumMME) / sumG;
//	curResult.fMOTAL = 1.0 - ((sumM + sumFP + std::log10(sumMME + 1))) / sumG;
//	curResult.fMissTargetPerGroundTruth = sumM / sumG;
//	curResult.fFalseAlarmPerGroundTruth = sumFP / sumG;
//	curResult.fRecall = sumC / sumG;
//	curResult.fPrecision = sumC / (sumFP + sumC);
//	curResult.fFalseAlarmPerFrame = sumFP / Fgt;
//
//	// MT PT ML
//	std::vector<int> MTstatsa(Ngt, 0);
//	curResult.nMostTracked = 0;
//	curResult.nPartilalyTracked = 0;
//	curResult.nMostLost = 0;
//	for (int i = 0; i < Ngt; i++)
//	{
//		double getLength = 0;
//		double trackedLength = 0;
//		int lastIndex;
//		for (int tIdx = 0; tIdx < Fgt; tIdx++)
//		{
//			if (0 != this->matXgt.at<double>(tIdx, i))
//			{
//				getLength++;
//				lastIndex = tIdx;
//				if (0 <= M.at<int>(tIdx, i)) { trackedLength++; }
//			}
//		}
//
//		if (trackedLength / getLength < 0.2)
//		{
//			MTstatsa[i] = 3;
//			curResult.nMostLost++;
//		}
//		else if (F >= lastIndex && trackedLength / getLength <= 0.8)
//		{
//			MTstatsa[i] = 2;
//			curResult.nPartilalyTracked++;
//		}
//		else if (trackedLength / getLength >= 0.8)
//		{
//			MTstatsa[i] = 1;
//			curResult.nMostTracked++;
//		}
//	}
//
//	// fragments
//	std::vector<int> fr(Ngt, 0);
//	curResult.nFragments = 0;
//	for (int i = 0; i < Ngt; i++)
//	{
//		int startIdx = 0;
//		int endIdx = 0;
//		int numSwtich = 0;
//		bool bTracked = false;
//		bool bStart = false;
//		for (int tIdx = 0; tIdx < Fgt; tIdx++)
//		{
//			if (0 <= M.at<int>(tIdx, i))
//			{
//				if (bStart) { endIdx = tIdx; }
//				else
//				{
//					bStart = true;					
//					startIdx = tIdx;
//				}
//				bTracked = true;
//			}
//			else
//			{
//				if (bTracked) {	numSwtich++; }
//				bTracked = false;
//			}
//		}
//		if (Fgt - 1 > endIdx) {	numSwtich--; }
//		curResult.nFragments += numSwtich;
//	}	
//
//	return curResult;
//}

void CEvaluator::PrintResultToConsole()
{
	printf("Evaluating PETS S2.L2 on ground plane...\n");
	printf("| Recl Prcn  FAR| MT PT ML|  FPR  FNR  FP  FN  ID  FM  err| MOTA MOTP MOTL\n");
	printf("|%5.1f%5.1f%5.2f|%3i%3i%3i|%5.1f%5.1f%4i%4i%4i%4i%5i|%5.1f %4.1f %4.1f\n", 
		this->m_stResult.fRecall * 100, 
		this->m_stResult.fPrecision * 100, 
		this->m_stResult.fFalseAlarmPerFrame, 
		this->m_stResult.nMostTracked, 
		this->m_stResult.nPartilalyTracked, 
		this->m_stResult.nMostLost, 
		this->m_stResult.fFalseAlarmPerGroundTruth * 100, 
		this->m_stResult.fMissTargetPerGroundTruth * 100,
		this->m_stResult.nFalsePositives, 
		this->m_stResult.nMissed, 
		this->m_stResult.nIDSwitch, 
		this->m_stResult.nFragments, 
		this->m_stResult.nMissed + this->m_stResult.nFalsePositives + this->m_stResult.nIDSwitch, 
		this->m_stResult.fMOTA * 100, 
		this->m_stResult.fMOTP * 100, 
		this->m_stResult.fMOTAL * 100);
}

void CEvaluator::PrintResultToFile(const char *strFilepathAndName)
{
	FILE *fp;
	try
	{
		fopen_s(&fp, strFilepathAndName, "w");
		fprintf_s(fp, "Evaluating PETS S2.L2 on ground plane...\n");
		fprintf_s(fp, "| Recl Prcn  FAR| MT PT ML|  FPR  FNR  FP  FN  ID  FM  err| MOTA MOTP MOTL\n");
		fprintf_s(fp, "|%5.1f%5.1f%5.2f|%3i%3i%3i|%5.1f%5.1f%4i%4i%4i%4i%5i|%5.1f %4.1f %4.1f\n", 
			this->m_stResult.fRecall * 100, 
			this->m_stResult.fPrecision * 100, 
			this->m_stResult.fFalseAlarmPerFrame, 
			this->m_stResult.nMostTracked, 
			this->m_stResult.nPartilalyTracked, 
			this->m_stResult.nMostLost, 
			this->m_stResult.fFalseAlarmPerGroundTruth * 100, 
			this->m_stResult.fMissTargetPerGroundTruth * 100,
			this->m_stResult.nFalsePositives, 
			this->m_stResult.nMissed, 
			this->m_stResult.nIDSwitch, 
			this->m_stResult.nFragments, 
			this->m_stResult.nMissed + this->m_stResult.nFalsePositives + this->m_stResult.nIDSwitch, 
			this->m_stResult.fMOTA * 100, 
			this->m_stResult.fMOTP * 100, 
			this->m_stResult.fMOTAL * 100);
		fclose(fp);
	}
	catch (DWORD dwError)
	{
		printf("[ERROR](PrintResultToFile) cannot open file! error code %d\n", dwError);
		return;
	}
}

void CEvaluator::PrintResultMatrix(const char *strFilepathAndName)
{
	FILE *fp;
	try
	{
		fopen_s(&fp, strFilepathAndName, "w");		
		
		// matX
		fprintf_s(fp, "MatX:(%d,%d)\n", this->matX.rows, this->matX.cols);
		for (int rowIdx = 0; rowIdx < this->matX.rows; rowIdx++)
		{
			for (int colIdx = 0; colIdx < this->matX.cols; colIdx++)
			{
				fprintf_s(fp, "%.6f,", this->matX.at<double>(rowIdx, colIdx));				
			}
			fprintf_s(fp , "\n");
		}

		// matY
		fprintf_s(fp, "MatY:(%d,%d)\n", this->matY.rows, this->matY.cols);
		for (int rowIdx = 0; rowIdx < this->matY.rows; rowIdx++)
		{
			for (int colIdx = 0; colIdx < this->matY.cols; colIdx++)
			{
				fprintf_s(fp, "%.6f,", this->matY.at<double>(rowIdx, colIdx));				
			}
			fprintf_s(fp , "\n");
		}
		fclose(fp);
	}
	catch (DWORD dwError)
	{
		printf("[ERROR](PrintResultMatrix) cannot open file! error code %d\n", dwError);
		return;
	}
}


