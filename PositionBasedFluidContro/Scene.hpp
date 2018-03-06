#ifndef SCENE_H
#define SCENE_H

#include "common.h"
#include "parameters.h"
#include "setupFunctions.h"

class Scene {
public:
	Scene(std::string name) : name(name) {}
	virtual void init(tempSolver* tp, solverParams* sp) = 0;

private:
	std::string name;

};

class ControlBase : public Scene {
public:
	ControlBase(std::string name) : Scene(name) {}

	virtual void init(tempSolver* tp, solverParams* sp) {
		const float radius = 0.1f;
		const float restDistance = radius * 0.5f;
		float3 lower = make_float3(0.0f, 0.1f, 0.0f);
		int3 dims = make_int3(120, 8, 120);
		createParticleGrid(tp, sp, lower, dims, restDistance);
		dims = make_int3(120, 120, 120);

		sp->radius = radius;
		sp->restDistance = restDistance;
		sp->numIterations = 2;
		sp->numDiffuse = 1024 * 2048;
		sp->numParticles = int(tp->positions.size());
		sp->numCloth = 0;
		sp->numConstraints = 0;
		sp->gravity = make_float3(0, -9.8f, 0);
		sp->bounds = make_float3(dims) * restDistance;
		sp->gridWidth = int(sp->bounds.x / restDistance);
		sp->gridHeight = int(sp->bounds.y / restDistance);
		sp->gridDepth = int(sp->bounds.z / restDistance);
		sp->gridSize = sp->gridWidth * sp->gridHeight * sp->gridDepth;

		sp->maxContacts = 10;
		sp->maxNeighbors = 50;
		sp->maxParticles = 50;
		sp->restDensity = 6378.0f;
		sp->lambdaEps = 600.0f;
		sp->vorticityEps = 0.0001f;
		sp->C = 0.0025f; //0.0025f;
		sp->K = 0.00001f;
		sp->KPOLY = 315.0f / (64.0f * PI * pow(radius, 9));
		sp->SPIKY = 45.0f / (PI * pow(radius, 6));
		sp->dqMag = 0.2f * radius;
		sp->wQH = sp->KPOLY * pow((radius * radius - sp->dqMag * sp->dqMag), 3);

		sp->numParticlesControl = int(tp->positionsControl.size());
		sp->maxNeighborsControl = 300;
		sp->maxParticlesControl = 50;
		sp->radiusControl = 0.22;
		sp->rgc = 4;
		sp->distD = 0.5;
		sp->Tlife = 3;
		sp->KPOLYControl = 315.0f / (64.0f * PI * pow(sp->radiusControl, 9));
		sp->SPIKYControl = 45.0f / (PI * pow(sp->radiusControl, 6));
		sp->logRedA = 15;
		sp->logRedB = 0.5;
		sp->alpha = 2.0;
		sp->beta =  2.0;
		sp->gamma = 0.00;
		sp->lambdaEpsControl = 40.0f;
		sp->restDensityControl = 3000.0f;

		sp->it = 0;

		tp->diffusePos.resize(sp->numDiffuse);
		tp->diffuseVelocities.resize(sp->numDiffuse);
	}
};


#endif
