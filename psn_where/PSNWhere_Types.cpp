#include "PSNWhere_Types.h"

/////////////////////////////////////////////////////////////////////////
// CTrackletCombination MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
CTrackletCombination::CTrackletCombination(void)
{
	memset(this->tracklets, NULL, sizeof(stTracklet2D*) * NUM_CAM);
	numTracklets = 0;
}

CTrackletCombination& CTrackletCombination::operator=(const CTrackletCombination &a)
{ 
	memcpy(this->tracklets, a.tracklets, sizeof(stTracklet2D*) * NUM_CAM);
	this->numTracklets = a.numTracklets;
	return *this;
}

bool CTrackletCombination::operator==(const CTrackletCombination &a)
{ 
	if (this->numTracklets != a.numTracklets) { return false; }
	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (this->tracklets[camIdx] != a.tracklets[camIdx])	{ return false; }
	}
	return true;
}

/************************************************************************
 Method Name: set
 Description: 
	- set the tracklet in the combination
 Input Arguments:
	- camIdx: target camera index
	- tracklet: target tracklet
 Return Values:
	- void
************************************************************************/
void CTrackletCombination::set(unsigned int camIdx, stTracklet2D *tracklet)
{ 
	if (this->tracklets[camIdx] == tracklet) { return; }
	if (NULL == this->tracklets[camIdx])
	{
		numTracklets++;
	}
	else if (NULL == tracklet)
	{
		numTracklets--;
	}
	this->tracklets[camIdx] = tracklet;
}

/************************************************************************
 Method Name: get
 Description: 
	- get the tracklet in the combination
 Input Arguments:
	- camIdx: target camera index
 Return Values:
	- tracklet of camIdx in the combination
************************************************************************/
stTracklet2D* CTrackletCombination::get(unsigned int camIdx)
{ 
	return this->tracklets[camIdx]; 
}

/************************************************************************
 Method Name: print
 Description: 
	- print current tracklet combination into the console
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CTrackletCombination::print()
{
	printf("[");
	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (NULL != tracklets[camIdx])
		{ 
			printf("%d", tracklets[camIdx]->id); 
		}
		else
		{ 
			printf("x"); 
		}

		if (camIdx < NUM_CAM - 1)
		{ 
			printf(","); 
		}
		else
		{ 
			printf("]\n"); 
		}
	}
}

/************************************************************************
 Method Name: checkCoupling
 Description: 
	- check wheather tracklets are compatible or not
 Input Arguments:
	- compCombination: the target of comparison
 Return Values:
	- true: incompatible/ false: compatible
************************************************************************/
bool CTrackletCombination::checkCoupling(CTrackletCombination &compCombination)
{
	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (this->tracklets[camIdx] == compCombination.tracklets[camIdx] && NULL != this->tracklets[camIdx])
		{
			return true;
		}
	}
	return false;
}

/************************************************************************
 Method Name: size
 Description: 
	- return the number of tracklets in the combination
 Input Arguments:
	- none
 Return Values:
	- tue number of tracklets in the combination
************************************************************************/
unsigned int CTrackletCombination::size()
{
	return this->numTracklets;
}

/////////////////////////////////////////////////////////////////////////
// CPointSmoother MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
CPointSmoother::CPointSmoother(void)
{
}

CPointSmoother::~CPointSmoother(void)
{
}

/************************************************************************
 Method Name: Insert
 Description: 
	- insert a single point into the smoothing sequence
 Input Arguments:
	- point: point being inserted
 Return Values:
	- the number of points that are modified by smoothing (from the back
	of the sequence)
************************************************************************/
int CPointSmoother::Insert(PSN_Point3D &point)
{
	// insertion
	int refreshPos = smootherX_.Insert(point.x);
	smootherY_.Insert(point.y);
	smootherZ_.Insert(point.z);

	// smoothing
	this->Update(refreshPos, 1);
	return refreshPos;
}

/************************************************************************
 Method Name: Insert
 Description: 
	- insert muliple points into the smoothing sequence
 Input Arguments:
	- points: points being inserted
 Return Values:
	- the number of points that are modified by smoothing (from the back
	of the sequence
************************************************************************/
int CPointSmoother::Insert(std::vector<PSN_Point3D> &points)
{
	// insertion
	std::vector<double> pointsX(points.size(), 0.0), pointsY(points.size(), 0.0), pointsZ(points.size(), 0.0);
	for (int pointIdx = 0; pointIdx < points.size(); pointIdx++)
	{
		pointsX[pointIdx] = points[pointIdx].x;
		pointsY[pointIdx] = points[pointIdx].y;
		pointsZ[pointIdx] = points[pointIdx].z;
	}
	int refreshPos = smootherX_.Insert(pointsX);
	smootherY_.Insert(pointsY);
	smootherZ_.Insert(pointsZ);

	// smoothing
	this->Update(refreshPos, (int)points.size());
	return refreshPos;
}

/************************************************************************
 Method Name: ReplaceBack
 Description: 
	- replace the last point of the sequence
 Input Arguments:
	- point: point for replacement
 Return Values:
	- the number of points that are modified by smoothing (from the back
	of the sequence
************************************************************************/
int CPointSmoother::ReplaceBack(PSN_Point3D &point)
{
	// replacement
	int refreshPos = smootherX_.ReplaceBack(point.x);
	smootherY_.ReplaceBack(point.y);
	smootherZ_.ReplaceBack(point.z);
	smoothedPoints_.pop_back();

	// smoothing
	this->Update(refreshPos, 1);
	return refreshPos;
}

/************************************************************************
 Method Name: SetQsets
 Description: 
	- Set precomputed Q sets for smoothing
 Input Arguments:
	- Qsets: precomputed Q sets
 Return Values:
	- none
************************************************************************/
void CPointSmoother::SetQsets(std::vector<Qset> *Qsets)
{
	smootherX_.SetPrecomputedQsets(Qsets);
	smootherY_.SetPrecomputedQsets(Qsets);
	smootherZ_.SetPrecomputedQsets(Qsets);
}

/************************************************************************
 Method Name: PopBack
 Description: 
	- pop the last point from smoothed sequence
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void CPointSmoother::PopBack(void)
{
	assert(0 < smoothedPoints_.size());
	smoothedPoints_.pop_back();
	smootherX_.PopBack();
	smootherY_.PopBack();
	smootherZ_.PopBack();	
}

/************************************************************************
 Method Name: SetSmoother
 Description: 
	- set the smoother by copying other smoother
 Input Arguments:
	- data: original sequence
	- smoothedPoints: smoothed sequence
	- span: window size of smoother
	- degree: degree for smoother
 Return Values:
	- none
************************************************************************/
void CPointSmoother::SetSmoother(std::deque<PSN_Point3D> &data, std::deque<PSN_Point3D> &smoothedPoints, int span, int degree)
{
	smoothedPoints_ = smoothedPoints;
	size_ = smoothedPoints_.size();
	back_ = &smoothedPoints_.back();
	std::deque<double> pointX, pointY, pointZ, smoothX, smoothY, smoothZ;
	for (int pos = 0; pos < smoothedPoints_.size(); pos++)
	{
		pointX.push_back(data[pos].x);
		pointY.push_back(data[pos].y);
		pointZ.push_back(data[pos].z);
		smoothX.push_back(smoothedPoints_[pos].x);
		smoothY.push_back(smoothedPoints_[pos].y);
		smoothZ.push_back(smoothedPoints_[pos].z);
	}
	smootherX_.SetSmoother(pointX, smoothX, span, degree);
	smootherY_.SetSmoother(pointY, smoothY, span, degree);
	smootherZ_.SetSmoother(pointZ, smoothZ, span, degree);
}

/************************************************************************
 Method Name: GetResult
 Description: 
	- get the specific point of smoothed sequence
 Input Arguments:
	- pos: the position of target point
 Return Values:
	- target point
************************************************************************/
PSN_Point3D CPointSmoother::GetResult(int pos)
{
	assert(pos < smoothedPoints_.size());
	return smoothedPoints_[pos];
}

/************************************************************************
 Method Name: GetResults
 Description: 
	- get points in the specific range of smoothed sequence
 Input Arguments:
	- startPos: starting position of the range
	- endPos: ending position of the range
 Return Values:
	- points sequence in the range
************************************************************************/
std::vector<PSN_Point3D> CPointSmoother::GetResults(int startPos, int endPos)
{
	if (endPos < 0) { endPos = (int)smoothedPoints_.size(); }
	assert(endPos <= smoothedPoints_.size() && startPos <= endPos);
	std::vector<PSN_Point3D> results;
	results.insert(results.end(), smoothedPoints_.begin() + startPos, smoothedPoints_.begin() + endPos);
	return results;
}

/************************************************************************
 Method Name: GetSmoother
 Description: 
	- backup the current smoother (by input arguments)
 Input Arguments:
	- data: backup of the original sequence
	- smoothedPoints: backup of the smoothed sequence
	- span: backup of current smoother's window size
	- degree: backup of the degree of current smoother
 Return Values:
	- none
************************************************************************/
void CPointSmoother::GetSmoother(std::deque<PSN_Point3D> &data, std::deque<PSN_Point3D> &smoothedPoints, int &span, int &degree)
{
	smoothedPoints = smoothedPoints_;
	std::deque<double> pointX, pointY, pointZ, smoothX, smoothY, smoothZ;
	smootherX_.GetSmoother(pointX, smoothX, span, degree);
	smootherY_.GetSmoother(pointY, smoothY, span, degree);
	smootherZ_.GetSmoother(pointZ, smoothZ, span, degree);
	data.resize(pointX.size());
	for (int pos = 0; pos < smoothedPoints_.size(); pos++)
	{
		data[pos].x = pointX[pos];
		data[pos].y = pointY[pos];
		data[pos].z = pointZ[pos];
	}
}

/************************************************************************
 Method Name: Update
 Description: 
	- do smoothing in the continuous sub interval of the sequence
 Input Arguments:
	- refresPos: the starting position of the interval
	- numPoints: the size of the interval
 Return Values:
	- none
************************************************************************/
void CPointSmoother::Update(int refreshPos, int numPoints)
{
	int endPos = (int)smoothedPoints_.size() + numPoints;
	smoothedPoints_.erase(smoothedPoints_.begin() + refreshPos, smoothedPoints_.end());
	for (int pos = refreshPos; pos < endPos; pos++)
	{
		smoothedPoints_.push_back(PSN_Point3D(smootherX_.GetResult(pos), smootherY_.GetResult(pos), smootherZ_.GetResult(pos)));
	}
	size_ = smoothedPoints_.size();
	back_ = &smoothedPoints_.back();
}

/////////////////////////////////////////////////////////////////////////
// Track3D MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
Track3D::Track3D()
	: id(0)
	, bActive(true)
	, bValid(true)
	, tree(NULL)
	, parentTrack(NULL)
	, timeStart(0)
	, timeEnd(0)
	, timeGeneration(0)
	, duration(1)
	, costTotal(0.0)
	, costReconstruction (0.0)
	, costLink(0.0)
	, costEnter(0.0)
	, costExit(0.0)
	, loglikelihood(0.0)
	, GTProb(0.0)
	, BranchGTProb(0.0)
	, bWasBestSolution(true)
	, bCurrentBestSolution(false)
	, bNewTrack(true)
{
}

Track3D::~Track3D()
{

}

/************************************************************************
 Method Name: Initialize
 Description: 
	- initialize the current track
 Input Arguments:
	- trackletCombination: current tracklet combination of the current track
	- id: its ID
	- timeGeneration: birth time of the track (not equal to starting time
	because it can be the child track, which is generated much later than
	its starting time)
	- parentTrack: its parent track. It can be NULL for seed tracks
 Return Values:
	- none
************************************************************************/
void Track3D::Initialize(CTrackletCombination &trackletCombination, unsigned int id, unsigned int timeGeneration, Track3D *parentTrack)
{
	this->id = id;
	this->curTracklet2Ds = trackletCombination;
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		this->timeTrackletEnded[camIdx] = 0;
		this->lastTrackletLocation3D[camIdx] = PSN_Point3D(0.0, 0.0, 0.0);
		this->lastTrackletSensitivity[camIdx] = 0;
	}
	this->numOutpoint = 0;
	this->timeEnd = timeGeneration;	
	this->timeGeneration = timeGeneration;
	if (NULL == parentTrack)
	{
		this->timeStart = timeGeneration;
		return;
	}	
	this->duration = this->timeEnd - this->timeStart + 1;
	this->tree = parentTrack->tree;
	this->parentTrack = parentTrack;
	this->timeStart = parentTrack->timeStart;
	this->costEnter = parentTrack->costEnter;
	this->loglikelihood = parentTrack->loglikelihood;
}

/************************************************************************
 Method Name: RemoveFromTree
 Description: 
	- remove track from the tree. children track's parent track will be
	modified.
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void Track3D::RemoveFromTree()
{
	// find valid parent (for children's adoption)
	Track3D *newParentTrack = this->parentTrack;
	while (NULL != newParentTrack && !newParentTrack->bValid) {	newParentTrack = newParentTrack->parentTrack; }

	// remove from children's parent pointer and adopt them to new parent
	for (std::deque<Track3D*>::iterator childTrackIter = this->childrenTrack.begin();
		childTrackIter != this->childrenTrack.end();
		childTrackIter++)
	{
		(*childTrackIter)->parentTrack = newParentTrack;
		if (NULL != (*childTrackIter)->parentTrack)
		{
			(*childTrackIter)->parentTrack->childrenTrack.push_back(*childTrackIter);
		}
	}

	// remove from parent's children list
	if (NULL != this->parentTrack)
	{
		for (std::deque<Track3D*>::iterator childTrackIter = this->parentTrack->childrenTrack.begin();
			childTrackIter != this->parentTrack->childrenTrack.end();
			childTrackIter++)
		{
			if ((*childTrackIter)->id != this->id) { continue; }
			this->parentTrack->childrenTrack.erase(childTrackIter);
			break;
		}
	}
}

//#define KALMAN_PROCESSNOISE_SIG (1.0E-5)
//#define KALMAN_MEASUREMENTNOISE_SIG (1.0E-5)
//#define KALMAN_POSTERROR_COV (0.1)
//#define KALMAN_CONFIDENCE_LEVEN (9)
//void Track3D::SetKalmanFilter(PSN_Point3D &initialPoint)
//{
//	this->KF.init(6, 3, 0);
//	this->KFMeasurement = cv::Mat(3, 1, CV_32FC1);
//
//	cv::setIdentity(this->KF.transitionMatrix); // [1,0,0,1,0,0; 0,1,0,0,1,0; 0,0,1,0,0,1; 0,0,0,1,0,0, ...]
//	this->KF.transitionMatrix.at<float>(0, 3) = 1.0f;
//	this->KF.transitionMatrix.at<float>(1, 4) = 1.0f;
//	this->KF.transitionMatrix.at<float>(2, 5) = 1.0f;
//
//	cv::setIdentity(this->KF.measurementMatrix);
//	cv::setIdentity(this->KF.processNoiseCov, cv::Scalar::all(KALMAN_PROCESSNOISE_SIG));
//	cv::setIdentity(this->KF.measurementNoiseCov, cv::Scalar::all(KALMAN_MEASUREMENTNOISE_SIG));
//	cv::setIdentity(this->KF.errorCovPost, cv::Scalar::all(KALMAN_POSTERROR_COV));
//
//	this->KF.statePost.at<float>(0, 0) = (float)initialPoint.x;
//	this->KF.statePost.at<float>(1, 0) = (float)initialPoint.y;
//	this->KF.statePost.at<float>(2, 0) = (float)initialPoint.z;
//	this->KF.statePost.at<float>(3, 0) = 0.0f;
//	this->KF.statePost.at<float>(4, 0) = 0.0f;
//	this->KF.statePost.at<float>(5, 0) = 0.0f;
//	cv::Mat curKFPrediction = this->KF.predict();
//}


/////////////////////////////////////////////////////////////////////////
// TrackTree MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
TrackTree::TrackTree()
	: id(0)
	, timeGeneration(0)	
	, bValid(true)
	, bConfirmed(false)
//	, numMeasurements(0)
//	, maxGTProb(0.0)
{
}

TrackTree::~TrackTree()
{
}

/************************************************************************
 Method Name: Initialize
 Description: 
	- initialize the current track tree with the seed track. also save the
	instance of the current track tree into the list container.
 Input Arguments:
	- seedTrack: seed track of the curren track tree
	- id: its ID
	- timeGeneration: track tree's birth time
	- treeList: saving container for instance of the current track tree
 Return Values:
	- 
************************************************************************/
void TrackTree::Initialize(Track3D *seedTrack, unsigned int id, unsigned int timeGeneration, std::list<TrackTree> &treeList)
{
	this->id = id;
	this->timeGeneration = timeGeneration;
	this->tracks.push_back(seedTrack);	
	treeList.push_back(*this);
	seedTrack->tree = &treeList.back();
}

/************************************************************************
 Method Name: ResetGlobalTrackProbInTree
 Description: 
	- reset GTP in the entire tracks in the current track tree
 Input Arguments:
	- none
 Return Values:
	- none
************************************************************************/
void TrackTree::ResetGlobalTrackProbInTree()
{
	for (std::deque<Track3D*>::iterator trackIter = this->tracks.begin();
		trackIter != this->tracks.end();
		trackIter++)
	{
		(*trackIter)->GTProb = 0.0;
	}
}

/************************************************************************
 Method Name: FindPruningPoint
 Description: 
	- find the lowest track in the branch that is the pruing point. children
	of it will be used in pruing
 Input Arguments:
	- timeWindowStart: where the pruning is starting
	- rootOfBranch: initial searching track. search is going to dive into the
	children of this track
 Return Values:
	- track which has the children will be condiered in the pruning step
************************************************************************/
Track3D* TrackTree::FindPruningPoint(unsigned int timeWindowStart, Track3D *rootOfBranch)
{
	if (NULL == rootOfBranch)
	{
		if (0 == this->tracks.size()) { return NULL; }
		rootOfBranch = this->tracks[0];
	}
	if (rootOfBranch->timeGeneration >= timeWindowStart) { return NULL; }

	// if more than one child placed inside of processing window, then the current node is the pruning point
	Track3D *curPruningPoint = NULL;
	for (PSN_TrackSet::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		curPruningPoint = FindPruningPoint(timeWindowStart, *trackIter);
		if (NULL == curPruningPoint) { break; }
	}

	if (NULL == curPruningPoint) { return rootOfBranch; }
	return curPruningPoint;
}

/************************************************************************
 Method Name: CheckBranchContainsBestSolution
 Description: 
	- check wheather the current branch has a track in the best solution or not
 Input Arguments:
	- rootOfBranch: the root of current branch
 Return Values:
	- wheather the current branch has a track in the best solution or not
************************************************************************/
bool TrackTree::CheckBranchContainsBestSolution(Track3D *rootOfBranch)
{
	if (rootOfBranch->bCurrentBestSolution) { return true; }

	for(PSN_TrackSet::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		if (CheckBranchContainsBestSolution(*trackIter)) { return true; }
	}
	return false;
}

/************************************************************************
 Method Name: SetValidityFlagInTrackBranch
 Description: 
	- Recursively set validity flags to 'bValid'
 Input Arguments:
	- rootOfBranch: the root of current branch
	- bValid: the value of validity flag
 Return Values:
	- none
************************************************************************/
void TrackTree::SetValidityFlagInTrackBranch(Track3D* rootOfBranch, bool bValid)
{
	rootOfBranch->bValid = bValid;
	for (std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		SetValidityFlagInTrackBranch(*trackIter, bValid);
	}
}

/************************************************************************
 Method Name: GetTracksInBranch
 Description: 
	- Recursively gather pointers of tracks in descendants
 Input Arguments:
	- rootOfBranch: the root of current branch
	- queueOutput: the pointer vector of tracks in the branch
 Return Values:
	- none
************************************************************************/
void TrackTree::GetTracksInBranch(Track3D* rootOfBranch, std::deque<Track3D*> &queueOutput)
{
	queueOutput.push_back(rootOfBranch);
	for (std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		GetTracksInBranch(*trackIter, queueOutput);
	}
}

/************************************************************************
 Method Name: MakeTreeNodesWithChildren
 Description: 
	- Recursively generate track node information
 Input Arguments:
	- queueChildrenTracks: queue of children tracks
	- parentNodeIdx: parent node index
	- outQueueNodes: output node index queue
 Return Values:
	- none
************************************************************************/
void TrackTree::MakeTreeNodesWithChildren(std::deque<Track3D*> queueChildrenTracks, const int parentNodeIdx, std::deque<stTrackInTreeInfo> &outQueueNodes)
{
	for (std::deque<Track3D*>::iterator trackIter = queueChildrenTracks.begin();
		trackIter != queueChildrenTracks.end();
		trackIter++)
	{
		stTrackInTreeInfo newInfo;
		newInfo.id = (*trackIter)->id;
		newInfo.parentNode = parentNodeIdx;
		newInfo.timeGenerated = (*trackIter)->timeGeneration;
		newInfo.GTP = (float)(*trackIter)->GTProb;
		//if ((*trackIter)->bValid) { outQueueNodes.push_back(newInfo); }
		outQueueNodes.push_back(newInfo);
		MakeTreeNodesWithChildren((*trackIter)->childrenTrack, (int)outQueueNodes.size(), outQueueNodes);
	}
}

/************************************************************************
 Method Name: GTProbOfBrach
 Description: 
	- Recursively sum global track probabilities of track branch
 Input Arguments:
	- rootOfBranch: the root of current branch
 Return Values:
	- sum of global track probability
************************************************************************/
double TrackTree::GTProbOfBrach(Track3D *rootOfBranch)
{
	double GTProb = rootOfBranch->GTProb;
	for (std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		GTProb += GTProbOfBrach(*trackIter);
	}
	return GTProb;
}

/************************************************************************
 Method Name: MaxGTProbOfBrach
 Description: 
	- find the maximum global track probability in the branch and set
	that value into 'BranchGTProb' property
 Input Arguments:
	- rootOfBranch: the seed track of the branch
 Return Values:
	- double: maximum value of global track probability in the branch
************************************************************************/
double TrackTree::MaxGTProbOfBrach(Track3D *rootOfBranch)
{
	double MaxGTProb = rootOfBranch->GTProb;
	for (std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		double curGTProb = MaxGTProbOfBrach(*trackIter);
		if (MaxGTProb < curGTProb) { MaxGTProb = curGTProb;	}
	}
	rootOfBranch->BranchGTProb = MaxGTProb;
	return MaxGTProb;
}

/************************************************************************
 Method Name: InvalidateBranchWithMinGTProb
 Description: 
	- invalidate the tracks in the branch which have smaller GTP then minGTProb
 Input Arguments:
	- rootOfBranch: the seed track of the branch
	- minGTProb: GTP threshold
 Return Values:
	- none
************************************************************************/
void TrackTree::InvalidateBranchWithMinGTProb(Track3D *rootOfBranch, double minGTProb)
{
	if (rootOfBranch->GTProb < minGTProb) { rootOfBranch->bValid = false; }
	for (std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		InvalidateBranchWithMinGTProb(*trackIter, minGTProb);
	}
}

/************************************************************************
 Method Name: FindMaxGTProbBranch
 Description: 
	- find the track having the maximum GTB int the branch
 Input Arguments:
	- queueChildrenTracks: queue of children tracks
 Return Values:
	- none
************************************************************************/
Track3D* TrackTree::FindMaxGTProbBranch(Track3D* branchSeedTrack, size_t timeIndex)
{
	if (branchSeedTrack->timeGeneration >= timeIndex) { return NULL; }

	Track3D* maxGTProbChild = NULL;
	for (std::deque<Track3D*>::iterator trackIter = branchSeedTrack->childrenTrack.begin();
		trackIter != branchSeedTrack->childrenTrack.end();
		trackIter++)
	{
		Track3D* curGTProbChild = FindMaxGTProbBranch((*trackIter), timeIndex);
		if (NULL == curGTProbChild){ continue; }
		if (NULL == maxGTProbChild || curGTProbChild > maxGTProbChild) { maxGTProbChild = curGTProbChild; }		
	}	
	return NULL == maxGTProbChild? branchSeedTrack : maxGTProbChild;
}

/************************************************************************
 Method Name: FindOldestTrackInBranch
 Description: 
	- find the oldest track in the branch
 Input Arguments:
	- 
 Return Values:
	- the oldest track in the branch
************************************************************************/
Track3D* TrackTree::FindOldestTrackInBranch(Track3D *trackInBranch, int nMostPreviousFrameIdx)
{
	Track3D *oldestTrack = trackInBranch;
	while (true) 
	{
		if (NULL == oldestTrack->parentTrack) { break; }
		if (nMostPreviousFrameIdx >= (int)oldestTrack->parentTrack->timeGeneration) { break; }
		oldestTrack = oldestTrack->parentTrack;
	}
	return oldestTrack;
}


ParamsAssociator3D::ParamsAssociator3D()
{
	nProcWindowSize_ = 10;
	nKBestSize_ = 1;
	nMaxTrackInOptimization_ = 1000;
	nMaxTrackInConfirmedTrackTree_ = 100;
	nMaxTrackInUnconfirmedTrackTree_ = 4;
	nNumFrameForConfirmation_ = 3;
	nMaxTimeJump_ = 9;
	fMaxMovingSpeed_ = 900.0;

	fProbEnterMax_ = 1.0E-1;
	fProbEnterDecayCoef_ = 1.0E-3;
	fCostEnterMax_ = 1000.0;

	fProbExitMax_ = 1.0E-2;
	fProbExitDecayCoef_dist_ = 1.0E-3;
	fProbExitDecayCoef_length_ = 1.0E-1;
	fCostExitMax_ = 1000.0;
}

ParamsAssociator3D::~ParamsAssociator3D()
{
}
