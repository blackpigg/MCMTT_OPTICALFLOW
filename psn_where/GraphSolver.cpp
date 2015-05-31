#include "GraphSolver.h"
#include <time.h>

// TODO: 티끌이라도 더 빠르게 하려면, delete(v) -> insert(u) 연계 시, OM 진입하는 OC는 u의 neighbor만 있음을 이용

#define PSN_GRAPH_MULTI_SOLUTION
//#define PSN_GRAPH_SOLVER_DEBUG
//#define PSN_GRAPH_TYPE_MWISP

const int nCurrentSolver = PSN_GRAPH_SOLVER_BLS;

// comparator
bool psnGraphComparatorVertexWeightDescend(const PSN_GraphVertex *vertex1, const PSN_GraphVertex *vertex2)
{
	return vertex1->weight > vertex2->weight;
}

bool psnGraphComparatorVertexDegreeDescend(const PSN_GraphVertex *vertex1, const PSN_GraphVertex *vertex2)
{
	return vertex1->queueNeighbor.size() > vertex2->queueNeighbor.size();
}

bool psnGraphComparatorVertexIDAscend(const PSN_GraphVertex *vertex1, const PSN_GraphVertex *vertex2)
{
	return vertex1->id < vertex2->id;
}

bool psnSolutionComparatorWeightSumDescend(const std::pair<PSN_VertexSet, double> &solution1, const std::pair<PSN_VertexSet, double> &solution2)
{
	return solution1.second > solution2.second;
}


/////////////////////////////////////////////////////////////////////////
// PSN_Graph
/////////////////////////////////////////////////////////////////////////
/* [[NOTICE]]: This graph data class has lazy update policy at deletion */
PSN_Graph::PSN_Graph(void)
	: m_nNumVertex(0)
	, m_nNumEdge(0)
	, m_nNewID(0)
{
}

PSN_Graph::PSN_Graph(size_t nNumVertex)
	: m_nNumVertex(nNumVertex)
	, m_nNumEdge(0)
	, m_nNewID(nNumVertex)
{
	for(size_t vertexIdx = 0; vertexIdx < nNumVertex; vertexIdx++)
	{
		this->m_listVertex.push_back(PSN_GraphVertex(vertexIdx));
	}
}

PSN_Graph::~PSN_Graph(void)
{
}

bool PSN_Graph::Clear()
{
	try
	{
		this->m_listVertex.clear();
		this->m_nNumVertex = 0;
		this->m_nNumEdge = 0;
		this->m_nNewID = 0;
		return true;
	}
	catch(DWORD dwError)
	{
		printf("[ERROR] Error is occured at Clear!!: %d\n", dwError);
	}
	return false;
}

bool PSN_Graph::TopologyModified()
{
	return this->m_bTopologyModified;
}

size_t PSN_Graph::maxDegree(void)
{
	size_t maxDegree = 0;
	for (std::list<PSN_GraphVertex>::iterator vertexIter = m_listVertex.begin();
		vertexIter != m_listVertex.end();
		vertexIter++)
	{
		maxDegree = std::max(maxDegree, vertexIter->degree());
	}
	return maxDegree;
}

size_t PSN_Graph::minDegree(void)
{
	size_t minDegree = 0;
	for (std::list<PSN_GraphVertex>::iterator vertexIter = m_listVertex.begin();
		vertexIter != m_listVertex.end();
		vertexIter++)
	{
		minDegree = std::min(minDegree, vertexIter->degree());
	}
	return minDegree;
}

double PSN_Graph::AverageVertexDegree(void)
{
	double averageDegree = 0;
	for (std::list<PSN_GraphVertex>::iterator vertexIter = m_listVertex.begin();
		vertexIter != m_listVertex.end();
		vertexIter++)
	{
		averageDegree += (double)vertexIter->degree();
	}
	averageDegree /= (double)m_listVertex.size();
	return averageDegree;
}

PSN_GraphVertex* PSN_Graph::AddVertex(double weight)
{
	this->m_listVertex.push_back(PSN_GraphVertex(this->m_nNewID++, weight));
	this->m_nNumVertex++;
	this->m_bTopologyModified = true;
	return &this->m_listVertex.back();
}

bool PSN_Graph::DeleteVertex(PSN_GraphVertex* vertex)
{
	vertex->valid = false;
	return true;
}

bool PSN_Graph::AddEdge(PSN_GraphVertex* vertex1, PSN_GraphVertex* vertex2)
{
	try
	{
		if(vertex1->valid && vertex2->valid)
		{
			vertex1->queueNeighbor.push_back(vertex2);
			vertex2->queueNeighbor.push_back(vertex1);
			this->m_bTopologyModified = true;
			m_nNumEdge++;
			return true;
		}
	}
	catch(DWORD dwError)
	{
		printf("[ERROR] Error is occured at AddEdge!!: %d\n", dwError);
	}
	return false;
}

bool PSN_Graph::Update(void)
{
	try
	{
		this->m_bTopologyModified = true;

		// delete invalid verteces from the neighbor list
		for(std::list<PSN_GraphVertex>::iterator vertexIter = this->m_listVertex.begin();
			vertexIter != this->m_listVertex.end();
			vertexIter++)
		{
			std::deque<PSN_GraphVertex*> newNeighbors;
			for(std::deque<PSN_GraphVertex*>::iterator neighborIter = (*vertexIter).queueNeighbor.begin();
				neighborIter != (*vertexIter).queueNeighbor.end();
				neighborIter++)
			{
				if((*neighborIter)->valid)
				{
					newNeighbors.push_back((*neighborIter));
				}
			}
			(*vertexIter).queueNeighbor = newNeighbors;
		}

		// delete invalid verteces
		for(std::list<PSN_GraphVertex>::iterator vertexIter = this->m_listVertex.begin();
			vertexIter != this->m_listVertex.end();
			/* do in the loop */)
		{
			if((*vertexIter).valid)
			{
				vertexIter++;
				continue;
			}
			this->m_listVertex.erase(vertexIter);
			this->m_nNumVertex--;
		}
	}
	catch(DWORD dwError)
	{
		printf("[ERROR] Error is occured at Update!!: %d\n", dwError);
		return false;
	}

	return true;
}

PSN_VertexSet PSN_Graph::GetAllVerteces(void)
{
	PSN_VertexSet allVertex;
	for(std::list<PSN_GraphVertex>::iterator vertexIter = this->m_listVertex.begin();
		vertexIter != this->m_listVertex.end();
		vertexIter++)
	{
		allVertex.push_back(&(*vertexIter));
	}

	return allVertex;
}

PSN_VertexSet PSN_Graph::GetNeighbors(PSN_GraphVertex* vertex)
{
	return vertex->queueNeighbor;
}

bool PSN_Graph::SetWeight(PSN_GraphVertex* vertex, double weight)
{
	try
	{
		if(vertex->valid)
		{
			vertex->weight = weight;
			return true;
		}
	}
	catch(DWORD dwError)
	{
		printf("[ERROR] Error is occured at SetWeight!!: %d\n", dwError);
	}
	return false;
}

double PSN_Graph::GetWeight(PSN_GraphVertex* vertex)
{
	try
	{
		if(vertex->valid)
		{
			return vertex->weight;
		}
	}
	catch(DWORD dwError)
	{
		printf("[ERROR] Error is occured at GetWeight!!: %d\n", dwError);
	}
	return 0.0;
}

/////////////////////////////////////////////////////////////////////////
// CGraphSolver CLASS MANAGEMENT
/////////////////////////////////////////////////////////////////////////
CGraphSolver::CGraphSolver(void)
	: m_pGraph(NULL)
	, m_bHasInitialSolution(false)
{
}

CGraphSolver::~CGraphSolver(void)
{
}

void CGraphSolver::Initialize(PSN_Graph* pGraph)
{
	this->m_pGraph = pGraph;
	this->m_bHasInitialSolution = false;

	// solver dependent initialization
	switch(nCurrentSolver)
	{
	case PSN_GRAPH_SOLVER_BLS:
		this->BLS_C.clear();
		this->BLS_PA.clear();
		this->BLS_OC.clear();
		this->BLS_OM.clear();
		this->BLS_nIter = 0;
		break;
	default:
		break;
	}
}

void CGraphSolver::Finalize(void)
{
	this->Clear();
}

void CGraphSolver::Clear(void)
{
	this->m_pGraph = NULL;

	// solver dependent finalization
	switch(nCurrentSolver)
	{
	case PSN_GRAPH_SOLVER_BLS:
		this->BLS_C.clear();
		this->BLS_PA.clear();
		this->BLS_OC.clear();
		this->BLS_OM.clear();
		break;
	default:
		break;
	}

	// clear initial solution
	this->m_queueInitialPoints.clear();
	this->m_bHasInitialSolution = false;

	// clear result
	this->m_stResult.solvingTime = 0.0;
	this->m_stResult.vecSolutions.clear();
}

void CGraphSolver::SetGraph(PSN_Graph* pGraph)
{
	this->Clear();
	this->m_pGraph = pGraph;
}

void CGraphSolver::SetInitialPoints(std::deque<PSN_VertexSet> initialPoints)
{
	if(0 == initialPoints.size()){ return; }

	this->m_queueInitialPoints = initialPoints;

	this->BLS_C.clear();
	this->BLS_PA.clear();
	this->BLS_OC.clear();
	this->BLS_OM.clear();

	// C
	for(PSN_VertexSet::iterator vertexIter = initialPoints.front().begin();
		vertexIter != initialPoints.front().end();
		vertexIter++)
	{
		this->BLS_C.push_back(*vertexIter);
		(*vertexIter)->CountUp();
	}

	// solution validation
#ifdef PSN_GRAPH_TYPE_MWISP
	size_t sizeCheckInC = 0;
#else	
	size_t sizeCheckInC = this->BLS_C.size();
#endif
	for(PSN_VertexSet::iterator vertexIter = initialPoints.front().begin();
		vertexIter != initialPoints.front().end();
		vertexIter++)
	{
		if(sizeCheckInC != (*vertexIter)->countNeighborInSolution)
		{
			// illegal solution
			this->m_queueInitialPoints.clear();
			this->BLS_C.clear();
			this->m_bHasInitialSolution = false;
			return;
		}
	}

	// PA and OC
#ifdef PSN_GRAPH_TYPE_MWISP
	size_t sizeOM = 1;
	size_t sizePA = 0;
#else
	size_t sizeOM = this->BLS_C.size() - 1;
	size_t sizePA = this->BLS_C.size();
#endif
	PSN_VertexSet allVertex = this->m_pGraph->GetAllVerteces();
	for(PSN_VertexSet::iterator vertexIter = allVertex.begin();
		vertexIter != allVertex.end();
		vertexIter++)
	{
		// OC
		if((*vertexIter)->bInSolution){ continue; }		
		this->BLS_OC.push_back(*vertexIter);

		// PA
		if(sizePA == (*vertexIter)->countNeighborInSolution)
		{
			this->BLS_PA.push_back(*vertexIter);
			continue;
		}

		//OM
		if(sizeOM != (*vertexIter)->countNeighborInSolution)
		{ continue;	}

#ifdef PSN_GRAPH_TYPE_MWISP
		// find pair
		for(PSN_VertexSet::iterator neighborIter = (*vertexIter)->queueNeighbor.begin();
			neighborIter != (*vertexIter)->queueNeighbor.end();
			neighborIter++)
		{
			if((*neighborIter)->bInSolution)
			{
				this->BLS_OM.push_back(std::make_pair(*vertexIter, *neighborIter));
				(*vertexIter)->bInOM = true;
				break;
			}
		}
#else
		// find another vertex
		PSN_VertexSet::iterator findIter;
		for(PSN_VertexSet::iterator vertexInCIter = this->BLS_C.begin();
			vertexInCIter != this->BLS_C.end();
			vertexInCIter++)
		{
			findIter = std::find((*vertexIter)->queueNeighbor.begin(), (*vertexIter)->queueNeighbor.end(), *vertexInCIter);
			if((*vertexIter)->queueNeighbor.end() != findIter)

			{ continue;	}

			this->BLS_OM.push_back(std::make_pair(*vertexIter, *vertexInCIter));
			(*vertexIter)->bInOM = true;
			break;
		}
#endif
	}

	this->m_bHasInitialSolution = true;
}


/////////////////////////////////////////////////////////////////////////
// GRAPH SOLVING
/////////////////////////////////////////////////////////////////////////
stGraphSolvingResult* CGraphSolver::Solve(void)
{
	if(NULL == this->m_pGraph)
	{
		return NULL;
	}

	time_t timerStart = clock();

	switch(nCurrentSolver)
	{
	case PSN_GRAPH_SOLVER_BLS:
		this->RunBLS();
		break;
	case PSN_GRAPH_SOLVER_ILS:
		this->RunILS();
		break;
	case PSN_GRAPH_SOLVER_AMTS:
		this->RunAMTS();
		break;
	case PSN_GRAPH_SOLVER_MCGA:
		this->RunMCGA();
		break;
	default:
		break;
	}

	time_t timerEnd = clock();
	this->m_stResult.solvingTime = (double)(timerEnd - timerStart) / CLOCKS_PER_SEC;

	return &(this->m_stResult);
}


//#define BLS_NUM_ITER (100000)
#define BLS_P0 (.75)
#define BLS_T (10)
#define BLS_PHI (7)
#define BLS_MIN_ITERATION (100)
#define BLS_MAX_ITERATION (2000)
void CGraphSolver::RunBLS(void)
{
	// solving information related
	time_t timer_start = clock();
	this->m_stResult.bestIteration = 0;
	this->m_stResult.maximumSearchInterval = 0;

	// parameters
	double L0 = 0.01 * (double)this->m_pGraph->Size();
	double Lmax = 0.1 * (double)this->m_pGraph->Size();
	double alphaS = 0.8;
	double alphaR = 0.8;

	/////////////////////////////////////////////////////////////////////////////
	// INITIAL SOLUTION
	/////////////////////////////////////////////////////////////////////////////
	double L = 0.0;
	double fc = 0.0;
	double fbest = 0.0;	
	size_t w = 0;
	PSN_VertexSet Cbest, Cp;
	
	if(this->m_bHasInitialSolution)
	{
		fc = 0.0;
		for(PSN_VertexSet::iterator vertexIter = this->BLS_C.begin();
			vertexIter != this->BLS_C.end();
			vertexIter++)
		{
			fc += (*vertexIter)->weight;
		}
	}
	else
	{
		fc = this->BLS_GenerateInitialSolution();
	}
	Cbest = this->BLS_C;
	fbest = fc;
	Cp = this->BLS_C;
	w = 0;

#ifdef PSN_GRAPH_MULTI_SOLUTION
	// save initial solution
	this->m_stResult.vecSolutions.push_back(std::make_pair(Cbest, fbest));
#endif

#ifdef PSN_GRAPH_TYPE_MWISP
	size_t maxIter = this->m_pGraph->Size() * 112;
#else
	size_t maxIter = this->m_pGraph->NumEdge() * 100;
#endif
	maxIter = std::min(std::max((size_t)BLS_MIN_ITERATION, maxIter), (size_t)BLS_MAX_ITERATION);
	this->m_stResult.iterationNumber = maxIter;

	/////////////////////////////////////////////////////////////////////////////
	// MAIN LOOP
	/////////////////////////////////////////////////////////////////////////////
	for(this->BLS_nIter = 0; this->BLS_nIter < maxIter; )
	{
		//------------------------------------------------------
		// LOCAL SEARCH
		//------------------------------------------------------
		double fIncrement = 0.0;
		while (true)
		{
			fIncrement = this->BLS_FindBestLocalMove();			
			if (0.0 >= fIncrement)
			{
				break;
			}
			fc += fIncrement;
			this->BLS_nIter++;
		}
		
		//------------------------------------------------------
		// SOLUTION CHECK
		//------------------------------------------------------
		fc = WeightSum(this->BLS_C);	// to handle precision error		
		if (fc > fbest)
		{
			Cbest = this->BLS_C;
			fbest = fc;
			w = 0;

			// solving info			
			this->m_stResult.maximumSearchInterval = std::max(this->m_stResult.maximumSearchInterval, (int)this->BLS_nIter - this->m_stResult.bestIteration);
			this->m_stResult.bestIteration = (int)this->BLS_nIter;
		}
		else
		{
			w++;
		}

		//------------------------------------------------------
		// PERTURBATION
		//------------------------------------------------------
		// sort for comparison
		std::sort(this->BLS_C.begin(), this->BLS_C.end(), psnGraphComparatorVertexIDAscend);
		if(w > BLS_T)
		{
			L = Lmax;
			w = 0;
		}
		else if(this->BLS_C == Cp)
		{
			L++;
		}
		else
		{
#ifdef PSN_GRAPH_MULTI_SOLUTION
			if(!CheckSolutionExistance(this->BLS_C, fc))
			{
				this->m_stResult.vecSolutions.push_back(std::make_pair(this->BLS_C, fc));
			}
#endif
			L = L0;
		}
		Cp = this->BLS_C;
		fc = this->BLS_Perturbation(L, w, alphaR, alphaS);
	}

	/////////////////////////////////////////////////////////////////////////////
	// TERMINATION
	/////////////////////////////////////////////////////////////////////////////
#ifdef PSN_GRAPH_MULTI_SOLUTION
	// sort the solution (descending order of weight sum)
	std::sort(this->m_stResult.vecSolutions.begin(), this->m_stResult.vecSolutions.end(), psnSolutionComparatorWeightSumDescend);
#else
	this->m_stResult.vecSolutions.push_back(std::make_pair(Cbest, fbest));
#endif
	std::sort(this->m_stResult.vecSolutions.front().first.begin(), this->m_stResult.vecSolutions.front().first.end(), psnGraphComparatorVertexIDAscend);	// by track ID

	// result packing
	time_t timer_end = clock();
	this->m_stResult.solvingTime = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
	this->m_stResult.numEdges = m_pGraph->NumEdge();
	this->m_stResult.numVertices = m_pGraph->Size();
	this->m_stResult.maximumDegree = m_pGraph->maxDegree();
	this->m_stResult.averageDegree = m_pGraph->AverageVertexDegree();
}

void CGraphSolver::RunILS(void)
{
}

void CGraphSolver::RunAMTS(void)
{
}

void CGraphSolver::RunMCGA(void)
{
}

#define PSN_GRAPH_SOLUTION_DUPLICATION_RESOLUTION (1.0E-5)
bool CGraphSolver::CheckSolutionExistance(PSN_VertexSet &vertexSet, double weightSum)
{
	for(size_t solutionIdx = 0; solutionIdx < this->m_stResult.vecSolutions.size(); solutionIdx++)
	{
		if(std::abs(weightSum - this->m_stResult.vecSolutions[solutionIdx].second) > PSN_GRAPH_SOLUTION_DUPLICATION_RESOLUTION)
		{
			continue;
		}
		if(vertexSet == this->m_stResult.vecSolutions[solutionIdx].first)
		{
			return true;
		}
	}
	return false;
}

double CGraphSolver::WeightSum(PSN_VertexSet &vertexSet)
{
	double weightSum = 0.0;
	for(PSN_VertexSet::iterator vertexIter = vertexSet.begin();
		vertexIter != vertexSet.end();
		vertexIter++)
	{
		weightSum += (*vertexIter)->weight;
	}
	return weightSum;
}


/////////////////////////////////////////////////////////////////////////
// BLS RELATED
/////////////////////////////////////////////////////////////////////////

/************************************************************************
 Method Name: BLS_GenerateInitialSolution
 Description: 
	- find initial solution of clique
 Input Arguments:
	- bMWISP: graph type
 Return Values:
	- double: objective value of solution set
************************************************************************/
double CGraphSolver::BLS_GenerateInitialSolution(void)
{
	this->BLS_C.clear();	// solution set
	this->BLS_PA.clear();	// outer vertex fully connected with C
	this->BLS_OM.clear();	// exclusive vertex pair set
	this->BLS_OC.clear();	// complement of the clique

	PSN_VertexSet allVertex = this->m_pGraph->GetAllVerteces();
	for(size_t insertIdx = 0; insertIdx < allVertex.size(); insertIdx++)
	{
		allVertex[insertIdx]->tabuStamp = 0;
		allVertex[insertIdx]->countNeighborInSolution = 0;
	}
	
	// construct C	
	double objectiveValue = 0.0;
	std::sort(allVertex.begin(), allVertex.end(), psnGraphComparatorVertexWeightDescend);
	for(PSN_VertexSet::iterator vertexIter = allVertex.begin();
		vertexIter != allVertex.end();
		vertexIter++)
	{
#ifdef PSN_GRAPH_TYPE_MWISP
		if(0 < (*vertexIter)->countNeighborInSolution)
#else
		if(this->BLS_C.size() > (*vertexIter)->countNeighborInSolution)
#endif
		{
			this->BLS_OC.push_back(*vertexIter);
			continue;
		}
		this->BLS_C.push_back(*vertexIter);
		objectiveValue += (*vertexIter)->weight;		

		// counting neighbors in C
		(*vertexIter)->CountUp();
	}

	// sort by track ID
	std::sort(this->BLS_C.begin(), this->BLS_C.end(), psnGraphComparatorVertexIDAscend);

	// construct OM
	for(PSN_VertexSet::iterator vertexIter = this->BLS_OC.begin();
		vertexIter != this->BLS_OC.end();
		vertexIter++)
	{
#ifdef PSN_GRAPH_TYPE_MWISP
		if(1 != (*vertexIter)->countNeighborInSolution)
		{ continue;	}

		// find pair
		for(PSN_VertexSet::iterator neighborIter = (*vertexIter)->queueNeighbor.begin();
			neighborIter != (*vertexIter)->queueNeighbor.end();
			neighborIter++)
		{
			if((*neighborIter)->bInSolution)
			{
				this->BLS_OM.push_back(std::make_pair(*vertexIter, *neighborIter));
				(*vertexIter)->bInOM = true;
				break;
			}
		}
#else
		if(this->BLS_C.size() - 1 != (*vertexIter)->countNeighborInSolution)
		{ continue;	}
		
		// find another vertex
		PSN_VertexSet::iterator findIter;
		for(PSN_VertexSet::iterator vertexInCIter = this->BLS_C.begin();
			vertexInCIter != this->BLS_C.end();
			vertexInCIter++)
		{
			findIter = std::find((*vertexIter)->queueNeighbor.begin(), (*vertexIter)->queueNeighbor.end(), *vertexInCIter);
			if((*vertexIter)->queueNeighbor.end() != findIter)

			{ continue;	}

			this->BLS_OM.push_back(std::make_pair(*vertexIter, *vertexInCIter));
			(*vertexIter)->bInOM = true;
			break;
		}
#endif
	}

#ifdef PSN_GRAPH_SOLVER_DEBUG
	this->PrintLog(4);
#endif

	return objectiveValue;
}


/************************************************************************
 Method Name: BLS_FindBestLocalMove
 Description: 
	- find the best local move
 Input Arguments:
	- none
 Return Values:
	- double: increment in objective value of solution set
************************************************************************/
double CGraphSolver::BLS_FindBestLocalMove(void)
{
	double maxObjectiveIncrement = 0.0;
	PSN_GraphVertex *vertexInsert = NULL;
	PSN_GraphVertex *vertexRemove = NULL;

	/////////////////////////////////////////////////////////////////////////////
	// SEARCH
	/////////////////////////////////////////////////////////////////////////////
	if(0 < this->BLS_PA.size())
	{
		std::sort(this->BLS_PA.begin(), this->BLS_PA.end(), psnGraphComparatorVertexWeightDescend);
		if(0.0 <= this->BLS_PA.front()->weight)
		{
			vertexInsert = this->BLS_PA.front();
			maxObjectiveIncrement = vertexInsert->weight;			
		}
	}

	// movement in OM
	for(size_t pairIdx = 0; pairIdx < this->BLS_OM.size(); pairIdx++)
	{
		if(this->BLS_OM[pairIdx].first->weight - this->BLS_OM[pairIdx].second->weight <= maxObjectiveIncrement)
		{
			continue;
		}
		vertexInsert = this->BLS_OM[pairIdx].first;
		vertexRemove = this->BLS_OM[pairIdx].second;
		maxObjectiveIncrement = this->BLS_OM[pairIdx].first->weight - this->BLS_OM[pairIdx].second->weight;
	}

	if(NULL == vertexInsert)
	{
		return 0.0;
	}

	// DEBUG
	//if(948 == this->BLS_nIter && 3 ==this->BLS_C.size() && this->BLS_C[0]->id == 16 && this->BLS_C[1]->id == 4 && this->BLS_C[2]->id == 15)
	//{
	//	int a = 0;
	//}

	// remove (remove -> insert is more effective than insert -> remove)
	this->BLS_VertexRemove(vertexRemove);

	// insert
	this->BLS_VertexInsert(vertexInsert);

#ifdef PSN_GRAPH_SOLVER_DEBUG
	this->PrintLog(0);
#endif
	return maxObjectiveIncrement;
}


/************************************************************************
 Method Name: BLS_Perturbation
 Description: 
	- select perturbation condition and do perturb
 Input Arguments:
	- L: perturbation strength
	- w: number of consecutive non-improving local optima visited
	- alphaR: coefficient
	- alphaS: coefficient
 Return Values:
	- double: objective value of solution set
************************************************************************/
const double BLS_wCheck = -BLS_T * std::log(BLS_P0);
double CGraphSolver::BLS_Perturbation(double L, size_t w, double alphaR, double alphaS)
{
	double fVal = 0.0;
	PSN_VertexSet newC;
	if(0 == w)
	{
		return this->BLS_PerturbRandom(L, alphaS);
	}
	
	double P = (double)w < BLS_wCheck? std::exp(-(double)w / (double)BLS_T) : BLS_P0;
	double unitRand = (double)rand() / (double)RAND_MAX;
	if(P >= unitRand)
	{
		return this->BLS_PerturbDirected(L);
	}
	return this->BLS_PerturbRandom(L, alphaR);
}


/************************************************************************
 Method Name: BLS_PerturbDirected
 Description: 
	- function for directed perturbation
 Input Arguments:
	- L: perturbation strength
 Return Values:
	- double: objective value of solution set
************************************************************************/
double CGraphSolver::BLS_PerturbDirected(double L)
{
	double fVal = 0.0;
	PSN_GraphVertex *vertexInsert = NULL;
	PSN_GraphVertex *vertexRemove = NULL;
	for(size_t LIdx = 0; LIdx < L; LIdx++)
	{		
		/////////////////////////////////////////////////////////////////////////////
		// UPDATE M AND SELECT MOVE
		/////////////////////////////////////////////////////////////////////////////
		fVal = 0.0;
		std::vector<std::pair<BLS_MOVE_TYPE, size_t>> queueMove;
		queueMove.reserve(this->BLS_PA.size() + this->BLS_OM.size() + this->BLS_C.size());
		for(size_t vertexIdx = 0; vertexIdx < this->BLS_PA.size(); vertexIdx++)
		{
			// insert from PA
			if(this->BLS_PA[vertexIdx]->tabuStamp <= this->BLS_nIter)
			{
				queueMove.push_back(std::make_pair(M1, vertexIdx));
			}			
		}
		for(size_t vertexIdx = 0; vertexIdx < this->BLS_OM.size(); vertexIdx++)
		{
			// switch
			if(this->BLS_OM[vertexIdx].first->tabuStamp <= this->BLS_nIter)
			{
				queueMove.push_back(std::make_pair(M2, vertexIdx));
			}			
		}
		for(size_t vertexIdx = 0; vertexIdx < this->BLS_C.size(); vertexIdx++)
		{
			// remove from solution set
			fVal += this->BLS_C[vertexIdx]->weight;
			queueMove.push_back(std::make_pair(M3, vertexIdx));
		}

		if(0 == queueMove.size())
		{
			this->BLS_nIter++;
			break;
		}

		// DEBUG
		//if(946 == this->BLS_nIter && 2 ==this->BLS_C.size() && this->BLS_C[0]->id == 1 && this->BLS_C[1]->id == 16)
		//{
		//	int a = 0;
		//}

		// random selection
		size_t selectedMoveIdx = (size_t)((double)(queueMove.size() - 1) * ((double)rand() / (double)RAND_MAX));		

		/////////////////////////////////////////////////////////////////////////////
		// MOVE
		/////////////////////////////////////////////////////////////////////////////		
		if(M1 == queueMove[selectedMoveIdx].first)
		{
			vertexInsert = this->BLS_PA[queueMove[selectedMoveIdx].second];
			vertexRemove = NULL;
		}
		else if(M2 == queueMove[selectedMoveIdx].first)
		{
			vertexInsert = this->BLS_OM[queueMove[selectedMoveIdx].second].first;
			vertexRemove = this->BLS_OM[queueMove[selectedMoveIdx].second].second;
		}
		else
		{
			vertexInsert = NULL;
			vertexRemove = this->BLS_C[queueMove[selectedMoveIdx].second];
		}

		// remove
		fVal += this->BLS_VertexRemove(vertexRemove);

		// insert
		fVal += this->BLS_VertexInsert(vertexInsert);

#ifdef PSN_GRAPH_SOLVER_DEBUG
		this->PrintLog(1);
#endif
		// increase iteration number
		this->BLS_nIter++;
	}

	return fVal;
}


/************************************************************************
 Method Name: BLS_PerturbRandom
 Description: 
	- function for random perturbation
 Input Arguments:
	- L: perturbation strength
	- alpha: coefficient
 Return Values:
	- double: objective value of solution set
************************************************************************/
double CGraphSolver::BLS_PerturbRandom(double L, double alpha)
{
	// calculate the objective value of current solution
	double fVal = WeightSum(this->BLS_C);
	PSN_GraphVertex *vertexInsert = NULL;
	for(size_t LIdx = 0; LIdx < L; LIdx++)
	{
		/////////////////////////////////////////////////////////////////////////////
		// UPDATE M AND SELECT MOVE
		/////////////////////////////////////////////////////////////////////////////
		std::vector<size_t> queueMove;
		queueMove.reserve(this->BLS_OC.size());
		
		for(size_t vertexIdx = 0; vertexIdx < this->BLS_OC.size(); vertexIdx++)
		{			
			if(this->BLS_OC[vertexIdx]->tabuStamp <= this->BLS_nIter)
			{
				queueMove.push_back(vertexIdx);
				continue;
			}
			
			double neighborWeightSumInC = 0.0;
			for(PSN_VertexSet::iterator vertexIter = this->BLS_OC[vertexIdx]->queueNeighbor.begin();
				vertexIter != this->BLS_OC[vertexIdx]->queueNeighbor.end();
				vertexIter++)
			{
				if((*vertexIter)->bInSolution)
				{
					neighborWeightSumInC += (*vertexIter)->weight;
				}
			}

			if(neighborWeightSumInC >= alpha * fVal)
			{
				queueMove.push_back(vertexIdx);
			}
		}

		if(0 == queueMove.size())
		{
			this->BLS_nIter++;
			break;
		}

		// random selection
		size_t selectedMoveIdx = (size_t)((double)(queueMove.size() - 1) * ((double)rand() / (double)RAND_MAX));

		/////////////////////////////////////////////////////////////////////////////
		// MOVE
		/////////////////////////////////////////////////////////////////////////////
		fVal = this->BLS_VertexInsertM4(selectedMoveIdx);

#ifdef PSN_GRAPH_SOLVER_DEBUG
		this->PrintLog(2);
#endif
		//// DEBUG
		//if(927 == this->BLS_nIter && 3 == this->BLS_C.size() && this->BLS_C[0]->id == 10 && this->BLS_C[1]->id == 15 && this->BLS_C[2]->id == 9)
		//{
		//	int a = 0;
		//}

		// increase iteration number
		this->BLS_nIter++;
	}

	return fVal;
}


/************************************************************************
 Method Name: BLS_VertexRemove
 Description: 
	- [NOTICE] Always be used with BLS_VertexInsert (for management of OM)
	- remove vertex from C and repair PA, OM and OC	
 Input Arguments:
	- vertexRemove: target vertex
 Return Values:
	- double: increment in objective value
************************************************************************/
double CGraphSolver::BLS_VertexRemove(PSN_GraphVertex *vertexRemove)
{
	if(NULL == vertexRemove)
	{ return 0.0; }
	
	// remove from C
	PSN_VertexSet::iterator removeIter = std::find(this->BLS_C.begin(), this->BLS_C.end(), vertexRemove);
	this->BLS_C.erase(removeIter);
	vertexRemove->CountDown();
	
	// insert to PA
	this->BLS_PA.push_back(vertexRemove);

	// insert to OC and update tabu_list
	this->BLS_OC.push_back(vertexRemove);
	vertexRemove->tabuStamp = this->BLS_GetMoveProhibitionNumber(this->BLS_nIter);

	if(0 == this->BLS_C.size())
	{
		this->BLS_PA = this->BLS_OC;
		this->BLS_OM.clear();
		return -vertexRemove->weight;
	}

	// update OM (-> OM and PA)
	std::deque<PSN_VertexPair> newOM;
	for(std::deque<PSN_VertexPair>::iterator pairIter = this->BLS_OM.begin();
		pairIter != this->BLS_OM.end();
		pairIter++)
	{
		if(vertexRemove != (*pairIter).second)
		{
			newOM.push_back(*pairIter);
			continue;
		}
		this->BLS_PA.push_back((*pairIter).first);
		(*pairIter).first->bInOM = false;
	}	

#ifdef PSN_GRAPH_TYPE_MWISP
	size_t sizeCheckOM = 1;
#else
	size_t sizeCheckOM = this->BLS_C.size() - 1;
#endif	

	// add to OM (<- non OM)
	for(PSN_VertexSet::iterator vertexIter = this->BLS_OC.begin();
		vertexIter != this->BLS_OC.end();
		vertexIter++)
	{
		if((*vertexIter)->bInOM || (*vertexIter)->countNeighborInSolution != sizeCheckOM)
		{ continue; }

		(*vertexIter)->bInOM = true;
//#ifdef PSN_GRAPH_SOLVER_DEBUG
//		printf("ADDDDDDDDDDDDDDDDDDD!!!!!!!!!!!! \n");
//#endif
#ifdef PSN_GRAPH_TYPE_MWISP
		for(PSN_VertexSet::iterator neighborIter = (*vertexIter)->queueNeighbor.begin();
			neighborIter != (*vertexIter)->queueNeighbor.end();
			neighborIter++)
		{
			if((*neighborIter)->bInSolution)
			{
				newOM.push_back(std::make_pair(*vertexIter, *neighborIter));				
				break;
			}
		}
#else
		PSN_VertexSet::iterator findIter;
		for(PSN_VertexSet::iterator solutionIter = this->BLS_C.begin();
			solutionIter != this->BLS_C.end();
			solutionIter++)
		{
			findIter = std::find((*vertexIter)->queueNeighbor.begin(), (*vertexIter)->queueNeighbor.end(), *solutionIter);
			if((*vertexIter)->queueNeighbor.end() == findIter)
			{
				newOM.push_back(std::make_pair(*vertexIter, *solutionIter));
				break;
			}
		}
#endif	
	}

	this->BLS_OM = newOM;
	newOM.clear();

	
	return -vertexRemove->weight;
}


/************************************************************************
 Method Name: BLS_VertexInsert
 Description: 
	- insert vertex to C by M1 or M2 and repair PA, OM and OC
 Input Arguments:
	- vertexInsert: target vertex
 Return Values:
	- double: increment in objective value
************************************************************************/
double CGraphSolver::BLS_VertexInsert(PSN_GraphVertex *vertexInsert)
{
	if(NULL == vertexInsert)
	{ return 0.0; }

	// remove from OC
	PSN_VertexSet::iterator removeIter = std::find(this->BLS_OC.begin(), this->BLS_OC.end(), vertexInsert);
	this->BLS_OC.erase(removeIter);

	// insert to C
	this->BLS_C.push_back(vertexInsert);
	vertexInsert->CountUp();

#ifdef PSN_GRAPH_TYPE_MWISP
	size_t sizeCheckPA = 0;
	size_t sizeCheckOM = 1;
#else
	size_t sizeCheckPA = this->BLS_C.size();
	size_t sizeCheckOM = this->BLS_C.size() - 1;
#endif	

	// update OM
	std::deque<PSN_VertexPair> newOM;
	for(std::deque<PSN_VertexPair>::iterator pairIter = this->BLS_OM.begin();
		pairIter != this->BLS_OM.end();
		pairIter++)
	{
		if(sizeCheckOM == (*pairIter).first->countNeighborInSolution)
		{
			newOM.push_back(*pairIter);
		}
	}
	this->BLS_OM = newOM;
	newOM.clear();

	// update PA
	PSN_VertexSet newPA;
	for(PSN_VertexSet::iterator vertexIter = this->BLS_PA.begin();
		vertexIter != this->BLS_PA.end();
		vertexIter++)
	{
		if(vertexInsert == *vertexIter)
		{ continue; }

		if(sizeCheckPA == (*vertexIter)->countNeighborInSolution)
		{
			newPA.push_back(*vertexIter);
		}
		else
		{
			this->BLS_OM.push_back(std::make_pair(*vertexIter, vertexInsert));
			(*vertexIter)->bInOM = true;
		}
	}
	this->BLS_PA = newPA;
	newPA.clear();

	return vertexInsert->weight;
}


/************************************************************************
 Method Name: BLS_VertexInsertM4
 Description: 
	- insert vertex to C by M4(from OC) and repair PA, OM and OC
 Input Arguments:
	- vertexIdxInOC: index of target vertex in OC
 Return Values:
	- double: objective value of solution set
************************************************************************/
double CGraphSolver::BLS_VertexInsertM4(size_t vertexIdxInOC)
{
	double fVal = 0.0;	

	/////////////////////////////////////////////////////////////////////////////
	// REPAIR C
	/////////////////////////////////////////////////////////////////////////////
	PSN_GraphVertex *vertexInsert = this->BLS_OC[vertexIdxInOC];
	for(PSN_VertexSet::iterator vertexIter = vertexInsert->queueNeighbor.begin();
		vertexIter != vertexInsert->queueNeighbor.end();
		vertexIter++)
	{
		(*vertexIter)->bInSolution = false;	// toggle for update C
	}

	// update C
	PSN_VertexSet newC;
	for(PSN_VertexSet::iterator vertexIter = this->BLS_C.begin();
		vertexIter != this->BLS_C.end();
		vertexIter++)
	{
#ifndef PSN_GRAPH_TYPE_MWISP
		(*vertexIter)->bInSolution = !(*vertexIter)->bInSolution;
#endif
		if((*vertexIter)->bInSolution)
		{
			newC.push_back(*vertexIter);
			fVal += (*vertexIter)->weight;
			continue;
		}
		(*vertexIter)->CountDown();		
		this->BLS_OC.push_back(*vertexIter);		
	}
	newC.push_back(vertexInsert);
	vertexInsert->CountUp();
	fVal += vertexInsert->weight;
	this->BLS_C = newC;

	// remove from OC
	this->BLS_OC.erase(this->BLS_OC.begin() + vertexIdxInOC);

#ifdef PSN_GRAPH_TYPE_MWISP
	size_t sizeCheckPA = 0;
	size_t sizeCheckOM = 1;
#else
	size_t sizeCheckPA = this->BLS_C.size();
	size_t sizeCheckOM = this->BLS_C.size() - 1;
#endif

	/////////////////////////////////////////////////////////////////////////////
	// REPAIR PA
	/////////////////////////////////////////////////////////////////////////////
	this->BLS_PA.clear();
	PSN_VertexSet OMCandidate;
	for(PSN_VertexSet::iterator vertexIter = this->BLS_OC.begin();
		vertexIter != this->BLS_OC.end();
		vertexIter++)
	{
		if(sizeCheckPA == (*vertexIter)->countNeighborInSolution	
			&& !(*vertexIter)->bInSolution)
		{
			this->BLS_PA.push_back(*vertexIter);
		}
		else if(sizeCheckOM == (*vertexIter)->countNeighborInSolution
			&& !(*vertexIter)->bInOM)
		{
			OMCandidate.push_back(*vertexIter);
		}		
	}


	/////////////////////////////////////////////////////////////////////////////
	// REPAIR OM
	/////////////////////////////////////////////////////////////////////////////
	std::deque<PSN_VertexPair> newOM;
	for(std::deque<PSN_VertexPair>::iterator pairIter = this->BLS_OM.begin();
		pairIter != this->BLS_OM.end();
		pairIter++)
	{
		if(sizeCheckOM != (*pairIter).first->countNeighborInSolution)
		{
			(*pairIter).first->bInOM = false;
			continue;
		}

		if((*pairIter).second->bInSolution)
		{
			newOM.push_back(*pairIter);			
		}
		else
		{
			newOM.push_back(std::make_pair((*pairIter).first, vertexInsert));
		}
	}

	// from OM candidate
	PSN_VertexSet::iterator findIter;
	for(PSN_VertexSet::iterator vertexIter = OMCandidate.begin();
		vertexIter != OMCandidate.end();
		vertexIter++)
	{
#ifdef PSN_GRAPH_TYPE_MWISP
		for(PSN_VertexSet::iterator neighborIter = (*vertexIter)->queueNeighbor.begin();
			neighborIter != (*vertexIter)->queueNeighbor.end();
			neighborIter++)
		{
			if((*neighborIter)->bInSolution)
			{
				newOM.push_back(std::make_pair(*vertexIter, *neighborIter));
				(*vertexIter)->bInOM = true;
				break;
			}
		}
#else
		for(PSN_VertexSet::iterator solutionVertexIter = this->BLS_C.begin();
			solutionVertexIter != this->BLS_C.end();
			solutionVertexIter++)
		{
			findIter = std::find((*vertexIter)->queueNeighbor.begin(), (*vertexIter)->queueNeighbor.end(), *solutionVertexIter);
			if((*vertexIter)->queueNeighbor.end() != findIter)
			{
				// found
				continue;
			}
			newOM.push_back(std::make_pair(*vertexIter, *solutionVertexIter));
			(*vertexIter)->bInOM = true;
			break;
		}
#endif
	}
	this->BLS_OM = newOM;

	return fVal;
}


/************************************************************************
 Method Name: BLS_GetMoveProhibitionNumber
 Description: 
	- get the ending iteration number of move prohibition
 Input Arguments:
	- sizeOM: size of OM set
	- nIter: current iteration
 Return Values:
	- size_t: ending iteration number of move prohibition
************************************************************************/
size_t CGraphSolver::BLS_GetMoveProhibitionNumber(size_t nIter)
{
	return nIter + BLS_PHI + (size_t)((double)this->BLS_OM.size() * ((double)rand() / (double)RAND_MAX));
}


// FOR DEBUG
void CGraphSolver::LoadGraphResult(size_t frameIdx, PSN_Graph *pGraph, std::deque<size_t> &solutionIdx, double &solvingTime)
{
	stGraphSolvingResult stResult;
	try
	{
		FILE *fp;
		char strFilePath[128];
#ifdef PSN_GRAPH_TYPE_MWISP
		sprintf_s(strFilePath, "%sGraphResult/%04d_MWISP.txt", RESULT_SAVE_PATH, frameIdx);
#else
		sprintf_s(strFilePath, "%sGraphResult/%04d.txt", RESULT_SAVE_PATH, frameIdx);
#endif
		fopen_s(&fp, strFilePath, "r");

		int tempInt = 0;
		fscanf_s(fp, "ID:%d\n", &tempInt);

		// vertex
		int numVertex = 0;
		float curWeight = 0.0;
		PSN_VertexSet setVertex;
		fscanf_s(fp, "vertex(id,weight):%d\n{", &numVertex);
		for(int vertexIdx = 0; vertexIdx < numVertex; vertexIdx++)
		{
			fscanf_s(fp, "(%d,%f),", &tempInt, &curWeight);
			setVertex.push_back(pGraph->AddVertex((double)curWeight));
		}
		
		// edge
		int edgeSource = 0;
		int edgeSink = 0;
		int numEdge = 0;
		fscanf_s(fp, "}\nedge(id,id):%d\n{", &numEdge);
		for(int vertexIdx = 0; vertexIdx < numEdge; vertexIdx++)
		{
			fscanf_s(fp, "(%d,%d),", &edgeSource, &edgeSink);
			pGraph->AddEdge(setVertex[edgeSource], setVertex[edgeSink]);
		}

		// solution
		int numVertexInSolution = 0;
		solutionIdx.clear();
		fscanf_s(fp, "}\nsolution:%d\n{", &numVertexInSolution);
		solutionIdx.resize(numVertexInSolution, 0);
		for(int vertexIdx = 0; vertexIdx < numVertexInSolution; vertexIdx++)
		{
			fscanf_s(fp, "%d,", &tempInt);
			solutionIdx[vertexIdx] = (size_t)tempInt;
		}

		// solving time
		float tempFloat = 0.0;
		fscanf_s(fp, "}\nsolvingTime:%f", &tempFloat);
		solvingTime = (double)tempFloat;

		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR] error is occured at \"PrintfGraphResult\": %d\n", dwError);
	}
}



void CGraphSolver::PrintLog(size_t type)
{
	printf("[iteration:%04d] ", this->BLS_nIter);
	switch(type)
	{
	case 0:
		printf("local search");
		break;
	case 1:
		printf("directed perturbation");
		break;
	case 2:
		printf("random perturbation");
		break;
	default:
		break;
	}
	printf("\n");

	// print C
	printf("C(%d)={", (int)this->BLS_C.size());
	for(size_t vertexIdx = 0; vertexIdx < this->BLS_C.size(); vertexIdx++)
	{
		if(0 < vertexIdx){ printf(","); }
		printf("%d", this->BLS_C[vertexIdx]->id);
	}
	printf("}\n");

	// print PA
	printf("PA(%d)={", (int)this->BLS_PA.size());
	for(size_t vertexIdx = 0; vertexIdx < this->BLS_PA.size(); vertexIdx++)
	{
		if(0 < vertexIdx){ printf(","); }
		printf("%d", this->BLS_PA[vertexIdx]->id);
	}
	printf("}\n");

	// print OM
	printf("OM(%d)={", (int)this->BLS_OM.size());
	for(size_t pairIdx = 0; pairIdx < this->BLS_OM.size(); pairIdx++)
	{
		if(0 < pairIdx){ printf(","); }
		printf("(%d,%d)", this->BLS_OM[pairIdx].first->id, this->BLS_OM[pairIdx].second->id);
	}
	printf("}\n");

	// print OC
	printf("OC(%d)={", (int)this->BLS_OC.size());
	for(size_t vertexIdx = 0; vertexIdx < this->BLS_OC.size(); vertexIdx++)
	{
		if(0 < vertexIdx){ printf(","); }
		printf("%d", this->BLS_OC[vertexIdx]->id);
	}
	printf("}\n");

	printf("\n");
}



//()()
//('')HAANJU.YOO
