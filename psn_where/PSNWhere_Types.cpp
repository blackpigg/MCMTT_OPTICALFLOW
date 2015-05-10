#define NOMINMAX // for preventing using window's minmax functions

#include "PSNWhere_Types.h"
#include "stdafx.h"

/////////////////////////////////////////////////////////////////////////
// CTrackletSet MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////

CTrackletSet::CTrackletSet(void)
{
	memset(this->tracklets, NULL, sizeof(stTracklet2D*) * NUM_CAM);
	numTracklets = 0;
}

CTrackletSet& CTrackletSet::operator=(const CTrackletSet &a)
{ 
	memcpy(this->tracklets, a.tracklets, sizeof(stTracklet2D*) * NUM_CAM);
	this->numTracklets = a.numTracklets;
	return *this;
}

bool CTrackletSet::operator==(const CTrackletSet &a)
{ 
	if (this->numTracklets != a.numTracklets) { return false; }
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(this->tracklets[camIdx] != a.tracklets[camIdx]) { return false; }
	}
	return true;
}

void CTrackletSet::set(unsigned int camIdx, stTracklet2D *tracklet)
{ 
	if (this->tracklets[camIdx] == tracklet) { return; }
	if(NULL == this->tracklets[camIdx])
	{
		numTracklets++;
	}
	else if(NULL == tracklet)
	{
		numTracklets--;
	}
	this->tracklets[camIdx] = tracklet;
}

stTracklet2D* CTrackletSet::get(unsigned int camIdx)
{ 
	return this->tracklets[camIdx]; 
}

void CTrackletSet::print()
{
	printf("[");
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(NULL != tracklets[camIdx]){ printf("%d", tracklets[camIdx]->id); }
		else{ printf("x"); }

		if(camIdx < NUM_CAM - 1){ printf(","); }
		else{ printf("]\n"); }
	}
}

bool CTrackletSet::checkCoupling(CTrackletSet &a)
{
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(this->tracklets[camIdx] == a.tracklets[camIdx] && NULL != this->tracklets[camIdx])
		{
			return true;
		}
	}
	return false;
}

unsigned int CTrackletSet::size()
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

int CPointSmoother::Insert(PSN_Point3D &point)
{
	int refreshPos = smootherX_.Insert(point.x);
	smootherY_.Insert(point.y);
	smootherZ_.Insert(point.z);

	// smoothing
	this->Update(refreshPos, 1);

	return refreshPos;
}

int CPointSmoother::Insert(std::vector<PSN_Point3D> &points)
{
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

PSN_Point3D CPointSmoother::GetResult(int pos)
{
	assert(pos < smoothedPoints_.size());
	return smoothedPoints_[pos];
}

std::vector<PSN_Point3D> CPointSmoother::GetResults(int startPos, int endPos)
{
	if (endPos < 0) { endPos = (int)smoothedPoints_.size(); }
	assert(endPos <= smoothedPoints_.size() && startPos <= endPos);
	std::vector<PSN_Point3D> results;
	results.insert(results.end(), smoothedPoints_.begin() + startPos, smoothedPoints_.begin() + endPos);
	return results;
}

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
	, bNewTrack(true)
	, bWasBestSolution(true)
	, bCurrentBestSolution(false)
	, tree(NULL)
	, parentTrack(parentTrack)
	, timeStart(0)
	, timeEnd(0)
	, timeGeneration(0)
	, duration(0)
	, costTotal(0.0)
	, costReconstruction (0.0)
	, costLink(0.0)
	, costEnter(0.0)
	, costExit(0.0)
	, costRGB(0.0)
	, loglikelihood(0.0)
	, GTProb(0.0)
{
}

Track3D::~Track3D()
{
}

void Track3D::Initialize(unsigned int id, Track3D *parentTrack, unsigned int timeGeneration, CTrackletSet &trackletSet)
{
	this->id = id;
	this->curTracklet2Ds = trackletSet;
	this->timeEnd = timeGeneration;
	for (int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if (NULL == trackletSet.get(camIdx)) { continue; }
		this->trackletLastLocation3D[camIdx] = trackletSet.get(camIdx)->currentLocation3D;
		this->timeTrackletEnded[camIdx] = timeGeneration;
		if (NULL == parentTrack)
		{
			this->tracklet2DIDs[camIdx].push_back(trackletSet.get(camIdx)->id);
		}
		else
		{
			this->tracklet2DIDs[camIdx] = parentTrack->tracklet2DIDs[camIdx];
			if (trackletSet.get(camIdx)->id != this->tracklet2DIDs[camIdx].back())
			{
				this->tracklet2DIDs[camIdx].push_back(trackletSet.get(camIdx)->id);
			}			
		}
		
	}
	if (NULL == parentTrack)
	{
		// new seed track		
		this->timeStart = timeGeneration;		
		this->duration = 1;
		return;
	}
	// branch track
	this->curTracklet2Ds = trackletSet;
	this->tree = parentTrack->tree;
	this->parentTrack = parentTrack;
	this->timeStart = parentTrack->timeStart;
	this->timeEnd = timeGeneration;
	this->timeGeneration = timeGeneration;
	this->duration = this->timeEnd - this->timeStart + 1;
	this->costTotal = parentTrack->costTotal;
	this->costReconstruction = parentTrack->costReconstruction;
	this->costLink = parentTrack->costLink;	
	this->costEnter = parentTrack->costEnter;
	this->costExit = 0.0;
	this->costRGB = 0.0;
	this->loglikelihood = parentTrack->loglikelihood;
	this->GTProb = parentTrack->GTProb;
}

int Track3D::InsertReconstruction(stReconstruction &reconstruction)
{
	this->reconstructions.push_back(reconstruction);
	stReconstruction *curReconstruction = &this->reconstructions.back();
	this->bActive = curReconstruction->bIsMeasurement;
	// smoothing
	int refreshStartPosition = pointSmoother.Insert(curReconstruction->point);
	for (int pos = refreshStartPosition; pos < this->reconstructions.size(); pos++)
	{
		this->reconstructions[pos].smoothedPoint = pointSmoother.GetResult(pos);
	}
	return refreshStartPosition;
}

void Track3D::RemoveFromTree()
{
	// find valid parent (for children's adoption)
	Track3D *newParentTrack = this->parentTrack;
	while(NULL != newParentTrack && !newParentTrack->bValid)
	{
		newParentTrack = newParentTrack->parentTrack;
	}

	// remove from children's parent pointer and adopt to new parent
	for(std::deque<Track3D*>::iterator childTrackIter = this->childrenTrack.begin();
		childTrackIter != this->childrenTrack.end();
		childTrackIter++)
	{
		(*childTrackIter)->parentTrack = newParentTrack;
		if(NULL != (*childTrackIter)->parentTrack)
		{
			(*childTrackIter)->parentTrack->childrenTrack.push_back(*childTrackIter);
		}
	}

	// remove from parent's children list
	if(NULL != this->parentTrack)
	{
		for(std::deque<Track3D*>::iterator childTrackIter = this->parentTrack->childrenTrack.begin();
			childTrackIter != this->parentTrack->childrenTrack.end();
			childTrackIter++)
		{
			if((*childTrackIter)->id != this->id)
			{ continue; }

			this->parentTrack->childrenTrack.erase(childTrackIter);
			break;
		}
	}
}

std::deque<Track3D*> Track3D::GatherValidChildrenTracks(Track3D* validParentTrack, std::deque<Track3D*> &targetChildrenTracks)
{
	PSN_TrackSet newChildrenTracks;
	for(PSN_TrackSet::iterator childTrackIter = targetChildrenTracks.begin();
		childTrackIter != targetChildrenTracks.end();
		childTrackIter++)
	{
		if((*childTrackIter)->bValid)
		{
			(*childTrackIter)->parentTrack = validParentTrack;
			newChildrenTracks.push_back(*childTrackIter);
			continue;
		}
		PSN_TrackSet foundChildrenTracks = GatherValidChildrenTracks(validParentTrack, (*childTrackIter)->childrenTrack);
		newChildrenTracks.insert(newChildrenTracks.end(), foundChildrenTracks.begin(), foundChildrenTracks.end());
	}

	return newChildrenTracks;
}

double Track3D::GetCost(void)
{
	return costTotal = costEnter + costReconstruction + costLink + costExit + costRGB;
}


/////////////////////////////////////////////////////////////////////////
// TrackTree MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
TrackTree::TrackTree()
	: id(0)
	, timeGeneration(0)
	, bValid(true)
	, numMeasurements(0)
//	, maxGTProb(0.0)
{
}

TrackTree::~TrackTree()
{
}

void TrackTree::Initialize(unsigned int id, Track3D *seedTrack, unsigned int timeGeneration, std::list<TrackTree> &treeList)
{
	this->id = id;
	this->timeGeneration = timeGeneration;
	this->tracks.push_back(seedTrack);
	
	treeList.push_back(*this);
	seedTrack->tree = &treeList.back();
	
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(NULL == seedTrack->curTracklet2Ds.get(camIdx))
		{
			continue;
		}
		stTracklet2DInfo newTrackletInfo;
		newTrackletInfo.tracklet2D = seedTrack->curTracklet2Ds.get(camIdx);
		newTrackletInfo.queueRelatedTracks.push_back(seedTrack);
		seedTrack->tree->tracklet2Ds[camIdx].push_back(newTrackletInfo);
		seedTrack->tree->numMeasurements++;
	}
	//this->m_queueActiveTrees.push_back(seedTrack->tree);
}

void TrackTree::InsertTrack(Track3D *track)
{
	tracks.push_back(track);
	track->tree = this;
}

/************************************************************************
 Method Name: ResetGlobalTrackProbInTree
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- none
************************************************************************/
void TrackTree::ResetGlobalTrackProbInTree()
{
	for(std::deque<Track3D*>::iterator trackIter = this->tracks.begin();
		trackIter != this->tracks.end();
		trackIter++)
	{
		(*trackIter)->GTProb = 0.0;
	}
}

Track3D* TrackTree::FindPruningPoint(unsigned int timeWindowStart, Track3D *rootOfBranch)
{
	if(NULL == rootOfBranch)
	{
		if(0 == this->tracks.size())
		{ return NULL; }
		rootOfBranch = this->tracks[0];
	}

	if(rootOfBranch->timeGeneration >= timeWindowStart)
	{ return NULL; }

	// if more than one child placed inside of processing window, then the current node is the pruning point
	Track3D *curPruningPoint = NULL;
	for(PSN_TrackSet::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		curPruningPoint = FindPruningPoint(timeWindowStart, *trackIter);
		if(NULL == curPruningPoint)
		{ break; }
	}

	if(NULL == curPruningPoint)
	{ return rootOfBranch; }

	return curPruningPoint;
}

bool TrackTree::CheckBranchContainsBestSolution(Track3D *rootOfBranch)
{
	if(rootOfBranch->bCurrentBestSolution)
	{ return true; }

	for(PSN_TrackSet::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		if(CheckBranchContainsBestSolution(*trackIter))
		{ return true; }
	}
	return false;
}

/************************************************************************
 Method Name: SetValidityFlagInTrackBranch
 Description: 
	- Recursively clear validity flags on descendants
 Input Arguments:
	- queueChildrenTracks: queue of children tracks
 Return Values:
	- none
************************************************************************/
void TrackTree::SetValidityFlagInTrackBranch(Track3D* rootOfBranch, bool bValid)
{
	rootOfBranch->bValid = bValid;
	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
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
	-
	-
 Return Values:
	- 
************************************************************************/
void TrackTree::GetTracksInBranch(Track3D* rootOfBranch, std::deque<Track3D*> &queueOutput)
{
	queueOutput.push_back(rootOfBranch);
	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
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
	for(std::deque<Track3D*>::iterator trackIter = queueChildrenTracks.begin();
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
	-
 Return Values:
	- double: sum of global track probability
************************************************************************/
double TrackTree::GTProbOfBrach(Track3D *rootOfBranch)
{
	double GTProb = rootOfBranch->GTProb;
	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		GTProb += GTProbOfBrach(*trackIter);
	}
	return GTProb;
}


///************************************************************************
// Method Name: MaxGTProbOfBrach
// Description: 
//	- find the maximum global track probability in the branch and set
//	that value into 'BranchGTProb' property
// Input Arguments:
//	- rootOfBranch: a pointer of the seed track of the branch
// Return Values:
//	- double: maximum value of global track probability in the branch
//************************************************************************/
//double TrackTree::MaxGTProbOfBrach(Track3D *rootOfBranch)
//{
//	double MaxGTProb = rootOfBranch->GTProb;
//	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
//		trackIter != rootOfBranch->childrenTrack.end();
//		trackIter++)
//	{
//		double curGTProb = MaxGTProbOfBrach(*trackIter);
//		if(MaxGTProb < curGTProb)
//		{
//			MaxGTProb = curGTProb;
//		}
//	}
//	rootOfBranch->BranchGTProb = MaxGTProb;
//	return MaxGTProb;
//}


/************************************************************************
 Method Name: InvalidateBranchWithMinGTProb
 Description: 
	- 
 Input Arguments:
	-
 Return Values:
	- none
************************************************************************/
void TrackTree::InvalidateBranchWithMinGTProb(Track3D *rootOfBranch, double minGTProb)
{
	if(rootOfBranch->GTProb < minGTProb)
	{
		rootOfBranch->bValid = false;
	}

	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		InvalidateBranchWithMinGTProb(*trackIter, minGTProb);
	}
}

/************************************************************************
 Method Name: FindMaxGTProbBranch
 Description: 
	- find 
 Input Arguments:
	- queueChildrenTracks: queue of children tracks
 Return Values:
	- none
************************************************************************/
Track3D* TrackTree::FindMaxGTProbBranch(Track3D* branchSeedTrack, size_t timeIndex)
{
	if(branchSeedTrack->timeGeneration >= timeIndex){ return NULL; }

	Track3D* maxGTProbChild = NULL;
	for(std::deque<Track3D*>::iterator trackIter = branchSeedTrack->childrenTrack.begin();
		trackIter != branchSeedTrack->childrenTrack.end();
		trackIter++)
	{
		Track3D* curGTProbChild = FindMaxGTProbBranch((*trackIter), timeIndex);
		if(NULL == curGTProbChild){ continue; }
		if(NULL == maxGTProbChild || curGTProbChild > maxGTProbChild)
		{
			maxGTProbChild = curGTProbChild;
		}		
	}	
	return NULL == maxGTProbChild? branchSeedTrack : maxGTProbChild;
}

/************************************************************************
 Method Name: FindOldestTrackInBranch
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- Track3D*: 
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

/************************************************************************
 Method Name: CheckConnectivityOfTrees
 Description: 
	- Check whether two trees share a common measurement or not
 Input Arguments:
	- tree1: 
	- tree2: 
 Return Values:
	- bool: true for connected trees
************************************************************************/
bool TrackTree::CheckConnectivityOfTrees(TrackTree *tree1, TrackTree *tree2)
{
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(0 == tree1->tracklet2Ds[camIdx].size() || 0 == tree2->tracklet2Ds[camIdx].size())
		{
			continue;
		}
		for(std::deque<stTracklet2DInfo>::iterator info1Iter = tree1->tracklet2Ds[camIdx].begin();
			info1Iter != tree1->tracklet2Ds[camIdx].end();
			info1Iter++)
		{
			for(std::deque<stTracklet2DInfo>::iterator info2Iter = tree2->tracklet2Ds[camIdx].begin();
				info2Iter != tree2->tracklet2Ds[camIdx].end();
				info2Iter++)
			{
				if((*info1Iter).tracklet2D == (*info2Iter).tracklet2D)
				{
					return true;
				}
			}
		}
	}
	return false;
}