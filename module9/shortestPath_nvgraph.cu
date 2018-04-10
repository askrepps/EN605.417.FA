/////////////////////////////
// shortestPath_nvgraph.cu // 
// Andrew Krepps           //
// Module 9 Assignment     //
// 4/9/2018                //
/////////////////////////////

#include <chrono>
#include <stdio.h>

#include <nvgraph.h>

#define NUM_VERTICES 6
#define NUM_VERTICES_PLUS_ONE 7
#define NUM_EDGES 9

///////////////////////////////////////////////////////////////////////////////////
//           finish         Example graph taken from                             //
//             v4           https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm //
//          9 /  \ 6                                                             //
//           /    \         Expected shortest path length from v0 to v4 is 20    //
//         v5      v3       (v0 -> v2 -> v5 -> v4)                               //
//          |\2 11/|                                                             //
//          | \  / |                                                             //
//        14|  v2  |15                                                           //
//          | /  \ |                                                             //
//          |/9 10\|                                                             //
//         v0------v1                                                            //
//      start   7                                                                //
///////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// define graph structure
	int destinationOffsetsData[NUM_VERTICES_PLUS_ONE] = {0, 0, 1, 3, 5, 7, NUM_EDGES};
	int sourceIndicesData[NUM_EDGES] = {0, 0, 1, 1, 2, 3, 5, 0, 2};
	float weights[NUM_EDGES] = {7.0f, 9.0f, 10.0f, 15.0f, 11.0f, 6.0f, 9.0f, 14.0f, 2.0f};
	int sourceVertex = 0;
	int destVertex = 4;
	float expectedPathLength = 20.0f;
	
	// initialize nvGRAPH
	nvgraphHandle_t handle;
	nvgraphCreate(&handle);
	
	// create and graph description
	nvgraphGraphDescr_t graph;
	nvgraphCreateGraphDescr(handle, &graph);
	
	// start clock
	auto start = std::chrono::high_resolution_clock::now();
	
	// initialize graph structure
	nvgraphCSCTopology32I_st cscInput;
	cscInput.nvertices = NUM_VERTICES;
	cscInput.nedges = NUM_EDGES;
	cscInput.destination_offsets = destinationOffsetsData;
	cscInput.source_indices = sourceIndicesData;
	nvgraphSetGraphStructure(handle, graph, (void*)&cscInput, NVGRAPH_CSC_32);
	
	// allocate weight and result data
	cudaDataType_t vertexDimT[1] = {CUDA_R_32F};
	cudaDataType_t edgeDimT[1] = {CUDA_R_32F};
	nvgraphAllocateVertexData(handle, graph, 1, vertexDimT);
	nvgraphAllocateEdgeData(handle, graph, 1, edgeDimT);
	nvgraphSetEdgeData(handle, graph, (void*)weights, 0);
	
	// run shortest path algorithm
	nvgraphSssp(handle, graph, 0, &sourceVertex, 0);
	
	// extract results	
	float allPathLengths[NUM_VERTICES];
	nvgraphGetVertexData(handle, graph, (void*)allPathLengths, 0);
	
	// stop clock and calculate exectuion time
	auto stop = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> duration(stop - start);
	float ms = duration.count()*1000.0f;
	
	// validate results
	float pathLength = allPathLengths[destVertex];
	printf("Path length from v0 -> v4: %f (execution time = %.6f ms)\n", pathLength, ms);
	if (pathLength != expectedPathLength) {
		printf("Error: path length (%f) did not match expected value (%f)\n", pathLength, expectedPathLength);
	}
	
	// clean up graph
	nvgraphDestroyGraphDescr(handle, graph);
	
	// terminate nvGRAPH
	nvgraphDestroy(handle);
	
	return EXIT_SUCCESS;
}
