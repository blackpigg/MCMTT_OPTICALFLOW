/******************************************************************************
 * Name : CGraphSolver
 * Date : 2014.08.31
 * Author : HAANJU.YOO
 * Version : 0.9
 * Description :
 *	class for graph and solver
 *
 ******************************************************************************/

#pragma once

#include "PSNWhere_Manager.h"


/////////////////////////////////////////////////////////////////////////
// DEFINES
/////////////////////////////////////////////////////////////////////////
#define PSN_GRAPH_SOLVER_BLS (0)
#define PSN_GRAPH_SOLVER_ILS (1)
#define PSN_GRAPH_SOLVER_AMTS (2)
#define PSN_GRAPH_SOLVER_MCGA (3)

/////////////////////////////////////////////////////////////////////////
// TYPEDEFS
/////////////////////////////////////////////////////////////////////////

// vertex
class PSN_GraphVertex
{
public:
	size_t id;
	bool valid;
	double weight;
	std::deque<PSN_GraphVertex*> queueNeighbor;

	// BLS related
	bool bInOM;
	bool bInSolution;
	size_t tabuStamp;
	size_t countNeighborInSolution;

	// constructors
	PSN_GraphVertex() : id(0), valid(true), weight(0.0), bInOM(false), bInSolution(false), tabuStamp(0), countNeighborInSolution(0) {}
	PSN_GraphVertex(size_t nID) : id(nID), valid(true), weight(0.0), bInOM(false), bInSolution(false),  tabuStamp(0), countNeighborInSolution(0) {}
	PSN_GraphVertex(size_t nID, double weight) : id(nID), valid(true), weight(weight), bInOM(false), bInSolution(false), tabuStamp(0), countNeighborInSolution(0) {}

	void CountUp(void)
	{
		this->bInSolution = true;
		this->bInOM = false;
		this->countNeighborInSolution++;
		for(std::deque<PSN_GraphVertex*>::iterator vertexIter = this->queueNeighbor.begin();
			vertexIter != this->queueNeighbor.end();
			vertexIter++)
		{
			(*vertexIter)->countNeighborInSolution++;
		}
	}

	void CountDown(void)
	{
		this->bInSolution = false;
		this->countNeighborInSolution--;
		for(std::deque<PSN_GraphVertex*>::iterator vertexIter = this->queueNeighbor.begin();
			vertexIter != this->queueNeighbor.end();
			vertexIter++)
		{
			(*vertexIter)->countNeighborInSolution--;
		}
	}

	size_t degree() const { return queueNeighbor.size(); }
};

// vertex set
typedef std::deque<PSN_GraphVertex*> PSN_VertexSet;

// edge
class PSN_GraphEdge
{
public:
	size_t id;
	bool valid;
	PSN_GraphVertex *vertex1;
	PSN_GraphVertex *vertex2;

	// constructors
	PSN_GraphEdge(PSN_GraphVertex *argVertex1, PSN_GraphVertex *argVertex2) : id(0), valid(true), vertex1(argVertex1), vertex2(argVertex2) {}
};

// graph
class PSN_Graph
{	
	//------------------------------------------------------
	// METHODS
	//------------------------------------------------------
public:
	PSN_Graph(void);
	PSN_Graph(size_t nNumVertex);
	~PSN_Graph(void);

	bool Clear(void);
	bool TopologyModified(void);
	size_t Size(void) const { return this->m_nNumVertex; }
	size_t NumEdge(void) const { return this->m_nNumEdge; }
	size_t maxDegree(void);
	size_t minDegree(void);
	double AverageVertexDegree(void);	

	PSN_GraphVertex* AddVertex(double weight);
	bool DeleteVertex(PSN_GraphVertex* vertex);
	bool AddEdge(PSN_GraphVertex* vertex1, PSN_GraphVertex* vertex2);
	bool Update(void);

	PSN_VertexSet GetAllVerteces(void);
	PSN_VertexSet GetNeighbors(PSN_GraphVertex* vertex);
	bool SetWeight(PSN_GraphVertex* vertex, double weight);
	double GetWeight(PSN_GraphVertex* vertex);

	//------------------------------------------------------
	// VARIABLES
	//------------------------------------------------------
private:
	bool m_bTopologyModified;
	size_t m_nNumVertex;
	size_t m_nNumEdge;
	size_t m_nNewID;
	std::list<PSN_GraphVertex> m_listVertex;
};

struct stGraphSolvingResult
{
	std::deque<std::pair<PSN_VertexSet, double>> vecSolutions;
	double solvingTime;
};
typedef std::pair<PSN_GraphVertex*, PSN_GraphVertex*> PSN_VertexPair;
typedef enum { M1 = 0, M2, M3, M4, A } BLS_MOVE_TYPE;



/////////////////////////////////////////////////////////////////////////
// GRAPH SOLVER CLASS DECLARATION
/////////////////////////////////////////////////////////////////////////
class CGraphSolver
{
	//------------------------------------------------------
	// VARIABLES
	//------------------------------------------------------
public:	

private:
	PSN_Graph *m_pGraph;
	bool m_bHasInitialSolution;
	std::deque<std::pair<PSN_VertexSet, double>> m_queueSolutions;
	std::deque<PSN_VertexSet> m_queueInitialPoints;
	stGraphSolvingResult m_stResult;

	// BLS related
	PSN_VertexSet BLS_C;
	PSN_VertexSet BLS_PA;
	PSN_VertexSet BLS_OC;
	std::deque<PSN_VertexPair> BLS_OM;
	size_t BLS_nIter;

	//------------------------------------------------------
	// METHODS
	//------------------------------------------------------
public:
	CGraphSolver(void);
	~CGraphSolver(void);

	void Initialize(PSN_Graph* pGraph);
	void Finalize(void);

	void Clear(void);
	void SetGraph(PSN_Graph* pGraph);
	void SetInitialPoints(std::deque<PSN_VertexSet> initialPoints);	

	stGraphSolvingResult* Solve(void);
	stGraphSolvingResult GetResult(void){ return this->m_stResult; };

	void LoadGraphResult(size_t frameIdx, PSN_Graph *pGraph, std::deque<size_t> &solutionIdx, double &solvingTime);
	void PrintLog(size_t type);

private:
	void RunBLS(void);
	void RunILS(void);
	void RunAMTS(void);
	void RunMCGA(void);

	// miscellaneous
	bool CheckSolutionExistance(PSN_VertexSet &vertexSet, double weightSum);
	double WeightSum(PSN_VertexSet &vertexSet);

	// BLS related
	double BLS_GenerateInitialSolution(void);
	double BLS_FindBestLocalMove(void);
	double BLS_Perturbation(double L, size_t w, double alphaR, double alphaS);
	double BLS_PerturbDirected(double L);
	double BLS_PerturbRandom(double L, double alpha);	
	double BLS_VertexRemove(PSN_GraphVertex *vertexRemove);
	double BLS_VertexInsert(PSN_GraphVertex *vertexInsert);	
	double BLS_VertexInsertM4(size_t vertexIdxInOC);	
	size_t BLS_GetMoveProhibitionNumber(size_t nIter);
};

//()()
//('')HAANJU.YOO
