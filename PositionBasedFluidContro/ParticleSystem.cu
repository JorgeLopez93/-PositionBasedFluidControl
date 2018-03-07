#ifndef PARTICLE_SYSTEM_CU
#define PARTICLE_SYSTEM_CU

#include "common.h"
#include "parameters.h"
#include "helper_math.h"

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }
static dim3 dims;
static dim3 dimsCon;
static dim3 diffuseDims;
static dim3 clothDims;
static dim3 gridDims;
static const int blockSize = 1024;

int it = 0;
bool flag_m = 1;

__constant__ solverParams sp;
__constant__ float deltaT = 0.0083f;
__device__ int foamCount = 0;
__constant__ float distr[] =
{
	-0.34828757091811f, -0.64246175794046f, -0.15712936555833f, -0.28922267225069f, 0.70090742209037f,
	0.54293139350737f, 0.86755128105523f, 0.68346917800767f, -0.74589352018474f, 0.39762042062246f,
	-0.70243115988673f, -0.85088539675385f, -0.25780126697281f, 0.61167922970451f, -0.8751634423971f,
	-0.12334015086449f, 0.10898816916579f, -0.97167591190509f, 0.89839695948101f, -0.71134930649369f,
	-0.33928178406287f, -0.27579196788175f, -0.5057460942798f, 0.2341509513716f, 0.97802030852904f,
	0.49743173248015f, -0.92212845381448f, 0.088328595779989f, -0.70214782175708f, -0.67050553191011f
};

__device__ float logReg(float x) {

	return 1/(1 + exp(sp.logRedA*(sp.logRedB - x) ) );
}

__device__ float WPoly6(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return 0;
	}

	return sp.KPOLY * pow((sp.radius * sp.radius - pow(rLen, 2)), 3);
}

__device__ float3 WSpiky(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return make_float3(0.0f);
	}

	float coeff = (sp.radius - rLen) * (sp.radius - rLen);
	coeff *= sp.SPIKY;
	coeff /= rLen;
	return r * -coeff;
}

__device__ float WPoly6Control(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radiusControl || rLen == 0) {
		return 0;
	}

	return sp.KPOLYControl * pow((sp.radiusControl * sp.radiusControl - pow(rLen, 2)), 3);
}

__device__ float3 WSpikyControl(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radiusControl || rLen == 0) {
		return make_float3(0.0f);
	}

	float coeff = (sp.radiusControl - rLen) * (sp.radiusControl - rLen);
	coeff *= sp.SPIKYControl;
	coeff /= rLen;
	return r * -coeff;
}

//Returns the eta vector that points in the direction of the corrective force
__device__ float3 eta(float4* newPos, int* neighbors, int* numNeighbors, int &index, float &vorticityMag) {
	float3 eta = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		eta += WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]])) * vorticityMag;
	}

	return eta;
}

__device__ float3 vorticityForce(float4* newPos, float3* velocities, int* neighbors, int* numNeighbors, int index) {
	//Calculate omega_i
	float3 omega = make_float3(0.0f);
	float3 velocityDiff;
	float3 gradient;

	for (int i = 0; i < numNeighbors[index]; i++) {
		velocityDiff = velocities[neighbors[(index * sp.maxNeighbors) + i]] - velocities[index];
		gradient = WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]));
		omega += cross(velocityDiff, gradient);
	}

	float omegaLength = length(omega);
	if (omegaLength == 0.0f) {
		//No direction for eta
		return make_float3(0.0f);
	}

	float3 etaVal = eta(newPos, neighbors, numNeighbors, index, omegaLength);
	if (etaVal.x == 0 && etaVal.y == 0 && etaVal.z == 0) {
		//Particle is isolated or net force is 0
		return make_float3(0.0f);
	}

	float3 n = normalize(etaVal);

	return (cross(n, omega) * sp.vorticityEps);
}

__device__ float sCorrCalc(float4 &pi, float4 &pj) {
	//Get Density from WPoly6
	float corr = WPoly6(make_float3(pi), make_float3(pj)) / sp.wQH;
	corr *= corr * corr * corr;
	return -sp.K * corr;
}

__device__ float3 xsphViscosity(float4* newPos, float3* velocities, int* neighbors, int* numNeighbors, int index) {
	float3 visc = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
			float3 velocityDiff = velocities[neighbors[(index * sp.maxNeighbors) + i]] - velocities[index];
			velocityDiff *= WPoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]));
			visc += velocityDiff;
	}

	return visc * sp.C;
}

__device__ void confineToBox(float4 &pos, float3 &vel) {
	if (pos.x < 0) {
		vel.x = 0;
		pos.x = 0.001f;
	} else if (pos.x > sp.bounds.x) {
		vel.x = 0;
		pos.x = sp.bounds.x - 0.001f;
	}

	if (pos.y < 0) {
		vel.y = 0;
		pos.y = 0.001f;
	} else if (pos.y > sp.bounds.y) {
		vel.y = 0;
		pos.y = sp.bounds.y - 0.001f;
	}

	if (pos.z < 0) {
		vel.z = 0;
		pos.z = 0.001f;
	} else if (pos.z > sp.bounds.z) {
		vel.z = 0;
		pos.z = sp.bounds.z - 0.001f;
	}
}

__device__ int3 getGridPos(float4 pos) {
	return make_int3(int(pos.x / sp.radius) % sp.gridWidth, int(pos.y / sp.radius) % sp.gridHeight, int(pos.z / sp.radius) % sp.gridDepth);
}

__device__ int getGridIndex(int3 pos) {
	return (pos.z * sp.gridHeight * sp.gridWidth) + (pos.y * sp.gridWidth) + pos.x;
}

__global__ void predictPositions(float4* newPos, float3* velocities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	//update velocity vi = vi + dt * fExt
	velocities[index] +=  sp.gravity * deltaT;

	//predict position x* = xi + dt * vi
	newPos[index] += make_float4(velocities[index] * deltaT, 0);

	confineToBox(newPos[index], velocities[index]);
}

__global__ void predictPositionsControl (float4* newPosControl, float3* velocityControl, float3 v) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticlesControl) return;
	velocityControl[index] = v;
	newPosControl[index] += make_float4(v * deltaT, 0);
	//printf("%d - ", index);
}

__global__ void clearNeighbors(int* numNeighbors, int* numContacts) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	numNeighbors[index] = 0;
	numContacts[index] = 0;
}

__global__ void clearNeighborsControl(int* numNeighborsControl) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticlesControl) return;

	numNeighborsControl[index] = 0;
}

__global__ void clearGrid(int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.gridSize) return;

	gridCounters[index] = 0;
}

__global__ void updateGrid(float4* newPos, int* gridCells, int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	int3 pos = getGridPos(newPos[index]);
	int gIndex = getGridIndex(pos);

	int i = atomicAdd(&gridCounters[gIndex], 1);
	i = min(i, sp.maxParticles - 1);
	gridCells[gIndex * sp.maxParticles + i] = index;
}

__global__ void clearGridControl(int* gridCountersControl) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.gridSize) return;

	gridCountersControl[index] = 0;
}

__global__ void updateNeighbors(float4* newPos, int* gridCells, int* gridCounters, int* neighbors, int* numNeighbors,
	int* contacts, int* numContacts)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	int3 pos = getGridPos(newPos[index]);
	int pIndex;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < sp.gridWidth && n.y >= 0 && n.y < sp.gridHeight && n.z >= 0 && n.z < sp.gridDepth) {
					int gIndex = getGridIndex(n);
					int cellParticles = min(gridCounters[gIndex], sp.maxParticles - 1);
					for (int i = 0; i < cellParticles; i++) {
						if (numNeighbors[index] >= sp.maxNeighbors) return;
						pIndex = gridCells[gIndex * sp.maxParticles + i];
						if (length(make_float3(newPos[index]) - make_float3(newPos[pIndex])) <= sp.radius) {
							neighbors[(index * sp.maxNeighbors) + numNeighbors[index]] = pIndex;
							numNeighbors[index]++;
						}
					}
				}

			}
		}
	}

}
__global__ void updateNeighborsControl(float4* newPos, float4* newPosControl, int* gridCells, int* gridCounters,
	  int* gridCellsControl, int* gridCountersControl,int* neighborsControl, int* numNeighborsControl)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticlesControl) return;

	int3 pos = getGridPos(newPosControl[index]);

	int pIndex;
	int gIndex = getGridIndex(pos);
	int id = atomicAdd(&gridCountersControl[gIndex], 1);
	id = min(id, sp.maxParticlesControl - 1);
	gridCellsControl[gIndex * sp.maxParticlesControl + id] = index;

	for (int z = -sp.rgc; z <= sp.rgc; z++) {
		for (int y = -sp.rgc; y <= sp.rgc; y++) {
			for (int x = -sp.rgc; x <= sp.rgc; x++) {
				int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < sp.gridWidth && n.y >= 0 && n.y < sp.gridHeight && n.z >= 0 && n.z < sp.gridDepth) {
					gIndex = getGridIndex(n);
					int cellParticles = min(gridCounters[gIndex], sp.maxParticles - 1);
					for (int i = 0; i < cellParticles; i++) {
						if (numNeighborsControl[index] >= sp.maxNeighborsControl) return;

						pIndex = gridCells[gIndex * sp.maxParticles + i];
						if (length(make_float3(newPosControl[index]) - make_float3(newPos[pIndex])) <= sp.radiusControl) {
							neighborsControl[(index * sp.maxNeighborsControl) + numNeighborsControl[index]] = pIndex;
							numNeighborsControl[index]++;
						}
					}
				}

			}
		}
	}

}

__global__ void updateNearestControl(float4* newPos, float4* newPosControl, int* gridCellsControl, int* gridCountersControl,
	 int* nearestControl, float* buffer0, float3* buffer1)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	int3 pos = getGridPos(newPos[index]);
	int pIndex;
	buffer0[index] = 0;
	buffer1[index] = make_float3(0);
	float r, r_m = sp.radiusControl;
	for (int z = -sp.rgc; z <= sp.rgc; z++) {
		for (int y = -sp.rgc; y <= sp.rgc; y++) {
			for (int x = -sp.rgc; x <= sp.rgc; x++) {
				int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < sp.gridWidth && n.y >= 0 && n.y < sp.gridHeight && n.z >= 0 && n.z < sp.gridDepth) {
					int gIndex = getGridIndex(n);
					int cellParticles = min(gridCountersControl[gIndex], sp.maxParticles - 1);
					for (int i = 0; i < cellParticles; i++) {
						//if (numNeighborsControl[index] >= sp.maxNeighborsControl) return;
						pIndex = gridCellsControl[gIndex * sp.maxParticlesControl + i];
						r = length(make_float3(newPosControl[pIndex]) - make_float3(newPos[index]));
						if (r < r_m) {
							r_m = r;
							nearestControl[index] = pIndex + 1;
						}
					}
				}

			}
		}
	}

}

__global__ void calcDensities(float4* newPos, int* neighbors, int* numNeighbors, float* densities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles ) return;

	float rhoSum = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		rhoSum += WPoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]));
	}

	densities[index] = rhoSum;
}

__global__ void calcDensitiesControl(float4* newPos, float4* newPosControl, int* neighborsControl,
	int* numNeighborsControl, float* densitiesControl, float* buffer0, float3* buffer1, float3* velocityControl)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticlesControl ) return;
	int nIndex;
	float rhoSum = 0.0f, rho;
	for (int i = 0; i < numNeighborsControl[index]; i++) {
		nIndex = neighborsControl[(index * sp.maxNeighborsControl) + i];
		rho = WPoly6Control(make_float3(newPosControl[index]), make_float3(newPos[nIndex]));
		rhoSum += rho;
		atomicAdd(&buffer0[nIndex], rho);
		atomicAdd(&buffer1[nIndex].x, rho * velocityControl[index].x);
		atomicAdd(&buffer1[nIndex].y, rho * velocityControl[index].y);
		atomicAdd(&buffer1[nIndex].z, rho * velocityControl[index].z);
	}
	//printf("%f , ", rhoSum);
	densitiesControl[index] = rhoSum;
}

__global__ void calcLambda(float4* newPos, int* neighbors, int* numNeighbors, float* densities, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles ) return;

	float densityConstraint = (densities[index] / sp.restDensity) - 1;
	float3 gradientI = make_float3(0.0f);
	float sumGradients = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {

		//Calculate gradient with respect to j
		float3 gradientJ = WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]])) / sp.restDensity;

		//Add magnitude squared to sum
		//sumGradients += pow(length(gradientJ), 2);
		sumGradients += pow(gradientJ.x, 2) + pow(gradientJ.y, 2) + pow(gradientJ.z, 2);
		gradientI += gradientJ;

	}
	//Add the particle i gradient magnitude squared to sum
	//sumGradients += pow(length(gradientI), 2);
	sumGradients += pow(gradientI.x, 2) + pow(gradientI.y, 2) + pow(gradientI.z, 2);
	buffer0[index] = (-1 * densityConstraint) / (sumGradients + sp.lambdaEps);
}

__global__ void calcLambdaControl(float4* newPos, float4* newPosControl, int* neighborsControl, int* numNeighborsControl,
	float* densitiesControl, float* buffer0)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticlesControl ) return;
	float densityConstraint = (densitiesControl[index] / sp.restDensityControl) - 1;
	//float3 gradientI = make_float3(0.0f);
	float sumGradients = 0.0f;
	for (int i = 0; i < numNeighborsControl[index]; i++) {
		//Calculate gradient with respect to j
		float3 gradientJ = WSpikyControl(make_float3(newPosControl[index]), make_float3(newPos[neighborsControl[(index * sp.maxNeighborsControl) + i]])) / sp.restDensityControl;
		//Add magnitude squared to sum
		//sumGradients += pow(length(gradientJ), 2);
		sumGradients += pow(gradientJ.x, 2) + pow(gradientJ.y, 2) + pow(gradientJ.z, 2);

	}
	buffer0[index] = (-1 * densityConstraint) / (sumGradients + sp.lambdaEpsControl);
	//printf("%f , ", buffer0[index]);
}

__global__ void updateReferences(float4* newPos, float4* newPosControl, int* nearestControl, int* referenceControl,
	int* timeLife, float3* deltaPs, float* buffer0, float3* buffer1, float3* velocities)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= sp.numParticles) return;

		int pIndex = nearestControl[index] - 1;
		if(pIndex > 0 && length(make_float3(newPosControl[pIndex]) - make_float3(newPos[index])) <=  sp.distD / 2){
			referenceControl[index] = pIndex;
			timeLife[index] = sp.Tlife;
		}
		if (timeLife[index] > 0) {
			timeLife[index]--;
			float l = length(make_float3(newPosControl[referenceControl[index]]) - make_float3(newPos[index]));
			deltaPs[index] = sp.beta*(logReg(l)/l)*(make_float3(newPosControl[referenceControl[index]]) - make_float3(newPos[index]));
		}else{
			deltaPs[index] = make_float3(0);
			referenceControl[index] = -1;
		}
		if (buffer0[index] != 0) {
			float3 delV = buffer1[index]/buffer0[index];
			deltaPs[index] += sp.gamma * deltaT * (delV - velocities[index]);
		}

	}

__global__ void calcDeltaP(float4* newPos, int* neighbors, int* numNeighbors, float3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;
	deltaPs[index] = make_float3(0);

	float3 deltaP = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		float lambdaSum = buffer0[index] + buffer0[neighbors[(index * sp.maxNeighbors) + i]];
		float sCorr = sCorrCalc(newPos[index], newPos[neighbors[(index * sp.maxNeighbors) + i]]);
		deltaP += WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]])) * (lambdaSum + sCorr);
	}
	deltaPs[index] = deltaP / sp.restDensity;
}

__global__ void calcDeltaPDensity(float4* newPos, float4* newPosControl, int* neighborsControl, int* numNeighborsControl,
	float3* deltaPs, float* buffer0)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticlesControl) return;

	float lambda =  -1 *  sp.alpha * buffer0[index] / sp.restDensityControl;
	float3 deltaP;
	int nIndex;

	//printf("%d , ", numNeighborsControl[index]);
	for (int i = 0; i < numNeighborsControl[index]; i++) {
		nIndex = neighborsControl[(index * sp.maxNeighborsControl) + i];
		deltaP = WSpikyControl(make_float3(newPosControl[index]), make_float3(newPos[nIndex])) ; //* (lambda);
		deltaP *= lambda;
		//float a = length(deltaP);
		//if(a > 0.01) printf("%f , ", a);
		atomicAdd(&deltaPs[nIndex].x, deltaP.x);
		atomicAdd(&deltaPs[nIndex].y, deltaP.y);
		atomicAdd(&deltaPs[nIndex].z, deltaP.z);
	}
}

__global__ void applyDeltaP(float4* newPos, float3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	newPos[index] += make_float4(deltaPs[index], 0);
	//newPos[index] += make_float4(deltaPs[index], 0);
}

__global__ void updateVelocities(float4* oldPos, float4* newPos, float3* velocities, int* neighbors, int* numNeighbors, float3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	//confineToBox(newPos[index], velocities[index]);

	//set new velocity vi = (x*i - xi) / dt
	velocities[index] = (make_float3(newPos[index]) - make_float3(oldPos[index])) / deltaT;

	//apply vorticity confinement
	velocities[index] += vorticityForce(newPos, velocities, neighbors, numNeighbors, index) * deltaT;

	//apply XSPH viscosity
	deltaPs[index] = xsphViscosity(newPos, velocities, neighbors, numNeighbors, index);

	//update position xi = x*i
	oldPos[index] = newPos[index];
}

__global__ void updateXSPHVelocities(float4* newPos, float3* velocities, float3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	velocities[index] += deltaPs[index] * deltaT;
}

__global__ void clearDeltaP(float3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	deltaPs[index] = make_float3(0);
	buffer0[index] = 0;
}


__global__ void updateSpheres(float4* oldPos, float3* spherePos, float3* spheres) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;
	long long int a;
	for (int i = 0; i < sp.numSpherePoints; i++) {
		a = (index * sp.numSpherePoints) + i;
		spheres[a].x = spherePos[i].x + oldPos[index].x - sp.bounds.x/2;
		spheres[a].y = spherePos[i].y + oldPos[index].y;// - sp.bounds.y/2;
		spheres[a].z = spherePos[i].z + oldPos[index].z - sp.bounds.z/2;
	}
}


void updateWater(solver* s, int numIterations) {
	//------------------WATER-----------------
	for (int i = 0; i < numIterations; i++) {
		//Calculate fluid densities and store in densities
		calcDensities<<<dims, blockSize>>>(s->newPos, s->neighbors, s->numNeighbors, s->densities);

		//Calculate all lambdas and store in buffer0
		calcLambda<<<dims, blockSize>>>(s->newPos, s->neighbors, s->numNeighbors, s->densities, s->buffer0);

		//calculate deltaP
		calcDeltaP<<<dims, blockSize>>>(s->newPos, s->neighbors, s->numNeighbors, s->deltaPs, s->buffer0);

		//update position x*i = x*i + deltaPi
		applyDeltaP<<<dims, blockSize>>>(s->newPos, s->deltaPs);
	}

	//Update velocity, apply vorticity confinement, apply xsph viscosity, update position
	updateVelocities<<<dims, blockSize>>>(s->oldPos, s->newPos, s->velocities, s->neighbors, s->numNeighbors,
	s->deltaPs);

	//Set new velocity
	updateXSPHVelocities<<<dims, blockSize>>>(s->newPos, s->velocities, s->deltaPs);

	updateSpheres<<<dims, blockSize>>>(s->oldPos, s->spherePos, s->spheres);

}


void update(solver* s, solverParams* sp) {


	if (it > 70 && it < 85) {
		predictPositionsControl<<<dimsCon, blockSize>>>(s->newPosControl, s->velocityControl, make_float3(0.0, 0.0 , 12.0));
		flag_m = 1;
		sp->gamma = 1.0;
	}else if (it > 90 && it < 120) {
		predictPositionsControl<<<dimsCon, blockSize>>>(s->newPosControl, s->velocityControl, make_float3(0.0, 0.0 , -12.0));
		flag_m = 1;
		sp->gamma = 1.0;
	}
	else if (it > 125 && it < 140) {
		predictPositionsControl<<<dimsCon, blockSize>>>(s->newPosControl, s->velocityControl, make_float3(0.0, 0.0 , 12.0));
		flag_m = 1;
		sp->gamma = 1.0;
	}else{
		sp->gamma = 0.2;
	}
	it ++;
	//printf("%d\n", it);
	//Predict positions and update velocity
	predictPositions<<<dims, blockSize>>>(s->newPos, s->velocities);

	//Update neighbors
	clearNeighbors<<<dims, blockSize>>>(s->numNeighbors, s->numContacts);
	clearGrid<<<gridDims, blockSize>>>(s->gridCounters);
	updateGrid<<<dims, blockSize>>>(s->newPos, s->gridCells, s->gridCounters);
	updateNeighbors<<<dims, blockSize>>>(s->newPos, s->gridCells, s->gridCounters, s->neighbors, s->numNeighbors,
	s->contacts, s->numContacts);

	if (flag_m) {
		clearGridControl<<<gridDims, blockSize>>>(s->gridCountersControl);
		clearNeighborsControl<<<dimsCon, blockSize>>>(s->numNeighborsControl);

		updateNeighborsControl<<<dimsCon, blockSize>>>(s->newPos, s->newPosControl, s->gridCells, s->gridCounters,
		s->gridCellsControl, s->gridCountersControl, s->neighborsControl, s->numNeighborsControl);

		flag_m = 1;
	}

	updateNearestControl<<<dims, blockSize>>>(s->newPos, s->newPosControl, s->gridCellsControl, s->gridCountersControl,
	s->nearestControl, s->buffer1, s->buffer2);

	calcDensitiesControl<<<dimsCon, blockSize>>>(s->newPos, s->newPosControl, s->neighborsControl,
	s->numNeighborsControl, s->densitiesControl, s->buffer1, s->buffer2, s->velocityControl);

	updateReferences<<<dims, blockSize>>>(s->newPos, s->newPosControl, s->nearestControl, s->referenceControl,
	s->timeLife, s->deltaPs, s->buffer1, s->buffer2, s->velocities);

	calcLambdaControl<<<dimsCon, blockSize>>>(s->newPos, s->newPosControl, s->neighborsControl,
	s->numNeighborsControl, s->densitiesControl, s->buffer1);

	calcDeltaPDensity<<<dimsCon, blockSize>>>(s->newPos, s->newPosControl, s->neighborsControl,
	s->numNeighborsControl, s->deltaPs, s->buffer1);

	applyDeltaP<<<dims, blockSize>>>(s->newPos, s->deltaPs);


	/*for (int i = 0; i < sp->numIterations; i++) {
		clearDeltaP<<<dims, blockSize>>>(s->deltaPs, s->buffer0);
		//particleCollisions<<<dims, blockSize>>>(s->newPos, s->contacts, s->numContacts, s->deltaPs, s->buffer0);
		applyDeltaP<<<dims, blockSize>>>(s->newPos, s->deltaPs, s->buffer0, 1);
	}*/

	//Solve constraints
	updateWater(s, sp->numIterations);
}

void setParams(solverParams *tempParams) {
	dims = int(ceil(tempParams->numParticles / blockSize + 0.5f));
	dimsCon = int(ceil(tempParams->numParticlesControl / blockSize + 0.5f));
	diffuseDims = int(ceil(tempParams->numDiffuse / blockSize + 0.5f));
	clothDims = int(ceil(tempParams->numConstraints / blockSize + 0.5f));
	gridDims = int(ceil(tempParams->gridSize / blockSize + 0.5f));
	cudaCheck(cudaMemcpyToSymbol(sp, tempParams, sizeof(solverParams)));
}

#endif
