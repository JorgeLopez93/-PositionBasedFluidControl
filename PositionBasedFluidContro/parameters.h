#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "common.h"

struct tempSolver {
	std::vector<float4> positions;
	std::vector<float3> velocities;
	std::vector<int> phases;
	std::vector<float3> spherePoints;
	std::vector<float4> positionsControl;

	std::vector<float4> diffusePos;
	std::vector<float3> diffuseVelocities;

	std::vector<int> clothIndices;
	std::vector<float> restLengths;
	std::vector<float> stiffness;
	std::vector<int> triangles;
};

struct solver {
	float3* spheres;
	float3* spherePos;

	float4* oldPos;
	float4* newPos;
	float3* velocities;

	int* phases;
	float* densities;


	float4* diffusePos;
	float3* diffuseVelocities;

	int* clothIndices;
	float* restLengths;
	float* stiffness;


	int* neighbors;
	int* numNeighbors;
	int* gridCells;
	int* gridCounters;
	int* contacts;
	int* numContacts;

	float3* deltaPs;

	float* buffer0;
	float* buffer1;
	float3* buffer2;

	float4* newPosControl;
	float3* velocityControl;
	float* densitiesControl;
	int* neighborsControl;
	int* numNeighborsControl;
	int* gridCountersControl;
	int* gridCellsControl;
	int* nearestControl;
	int* referenceControl;
	int* timeLife;
};

struct solverParams {
public:
	int maxNeighbors;
	int maxParticles;
	int maxContacts;
	int gridWidth, gridHeight, gridDepth;
	int gridSize;
	int it;

	int numParticles;
	int numDiffuse;
	int numSpherePoints;

	int numCloth;
	int numConstraints;


	float3 gravity;
	float3 bounds;

	int numIterations;
	float radius;
	float restDistance;
	//float sor;
	//float vorticity;

	float KPOLY;
	float SPIKY;
	float restDensity;
	float lambdaEps;
	float vorticityEps;
	float C;
	float K;
	float dqMag;
	float wQH;

	int maxNeighborsControl;
	int maxParticlesControl;
	int numParticlesControl;
	float radiusControl;
	int rgc;
	float distD;
	int Tlife;

	float KPOLYControl;
	float SPIKYControl;
	float logRedA;
	float logRedB;
	float alpha;
	float beta;
	float gamma;
	float lambdaEpsControl;
	float restDensityControl;
};

#endif
