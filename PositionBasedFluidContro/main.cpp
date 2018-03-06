#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h>
#include <vector>
#include <math.h>
#include <cfloat>
#include "GL/glew.h"
#include "GL/glut.h"
#include "Angel-yjc.h"
#include "stb_image.c"
#include <cuda_gl_interop.h>
#include "ParticleSystem.cpp"
#include "Scene.hpp"
#include "common.h"

using namespace std;

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }

#define WIDTH  600
#define HEIGHT 600


static const GLfloat lastX = (600 / 2);
static const GLfloat lastY = (600 / 2);
static float deltaTime = 0.0f;
static float lastFrame = 0.0f;
static int w = 0;
static bool ss = false;

cudaGraphicsResource *resources1, *resources;
cudaArray            *cuda_image_array;



GLuint Angel::InitShader(const char* vShaderFile, const char* fShaderFile);
void reducTri(int p1, int p2, int p3);

GLuint cube_buffer, grid_buffer, grid_bufferAtrip;

static GLuint texName, textureID;

mat4 rot = Rotate(0, 1, 1, 1);

std::vector<vec3> cube_vertices;

vec3 verticesRay[8]={
  vec3(-0.5f,-0.5f,-0.5f),
	vec3( 0.5f,-0.5f,-0.5f),
	vec3( 0.5f, 0.5f,-0.5f),
	vec3(-0.5f, 0.5f,-0.5f),

	vec3(-0.5f,-0.5f, 0.5f),
	vec3( 0.5f,-0.5f, 0.5f),
	vec3( 0.5f, 0.5f, 0.5f),
	vec3(-0.5f, 0.5f, 0.5f)
};

GLushort cubeIndicesRay[36]={
  0,5,4,
  5,0,1,
  3,7,6,
  3,6,2,
  7,4,6,
  6,4,5,
  2,1,3,
  3,1,0,
  3,0,7,
  7,0,4,
  6,5,2,
  2,5,1
};
GLushort cube_indices[] = {
  0, 1, 2, 3,
  3, 2, 6, 7,
  7, 6, 5, 4,
  4, 5, 1, 0,
  0, 3, 7, 4,
  1, 2, 6, 5,
};

typedef Angel::vec4  color4;
typedef Angel::vec3  point3;
typedef Angel::vec4  vec4;


static int value = 0;

std::vector<point3> gridMap;
std::vector<point3> gridMapNormal;
std::vector<point3> gridMapColor;
std::vector<vec2> gridMapTexCoordr;

std::vector<point3> sphere_points;
std::vector<point3> sphere_norm;

std::vector<point3> spheres_control;
std::vector<point3> spheres_norm_control;
std::vector<point3> spheres_color_control;

std::vector<int> indMap;
std::vector<int> indMapNormal;
std::vector<int> indMapTexCoordr;


int numNodes, widthTex, heightTex, componentsTex;
unsigned char *texColor;

GLuint program;
GLuint program2;
bool flag_gird, firstMouse = 1;
float yaw, pitch;
vec4 cameraFront;

GLfloat  fovy = 20.0;
GLfloat  aspect;
GLfloat  zNear = .2, zFar = 5.0;

vec4 VRP = vec4(-3.0, 0.0, 0.0, 1.0);
vec4 VPN = vec4(1.0, 0.0, 0.0, 0.0);
vec4 VUP = vec4(0.0, 1.0, 0.0, 0.0);
vec4 eye = vec4(0.0, 0.0, 0.0, 0.0);

float R = 0.1, E = 2.5, mx, my, mz, Mx, My, Mz;


void readSphere(string s){
  freopen(s.c_str(),"r", stdin);
  int n, d, i, j;
  float x, y, z;
  cin>> n;
  for (i = 0; i < n; i++)  {
    cin>> d;
    for (j = 0; j < d; j++) {
      cin>> x >> y >> z;
      sphere_points.push_back(point3( x, y, z)*((R)/2));
      sphere_norm.push_back(normalize(point3( x, y, z)));

    }
  }
  fclose (stdin);
  numNodes = sphere_points.size();
  std::cout << numNodes << '\n';
}

void readFace(string a){
  std::vector<int> v = {0, 0, 0};
  char c;
  int j = 0, i;
  for (i = 0; i < a.size(); i++) {
    c = a[i];
    if (c == '/') {
      j++;
    } else {
      v[j] *= 10;
      v[j] += (c - '0');
    }
  }
  indMap.push_back(v[0] - 1);
  indMapTexCoordr.push_back(v[1] - 1);
  indMapNormal.push_back(v[2] - 1);
}

void readObje(string s){
  mx = FLT_MAX; my = FLT_MAX; my = FLT_MAX;
  Mx = -FLT_MAX; My = -FLT_MAX; My = -FLT_MAX;
  freopen(s.c_str(),"r", stdin);
  float x, y, z;
  string t;
  cin>> t;
  while (t == "v") {
    cin>> x >> y >> z;
    x *= E; y *= E; z *= E;
    gridMap.push_back(point3( x, y, z));
    if(mx > x) mx = x;
    if(my > y) my = y;
    if(mz > z) mz = z;

    if(Mx < x) Mx = x;
    if(My < y) My = y;
    if(Mz < z) Mz = z;
    cin>> t;
  }
  while (t == "vt") {
    cin>> x >> y;
    gridMapTexCoordr.push_back(vec2( x, y));
    cin>> t;
  }
  while (t == "vn") {
    cin>> x >> y >> z;
    gridMapNormal.push_back(point3( x, y, z));
    cin>> t;
  }
  while (t != "f") {
    cin>> t;
  }
  while (t == "f") {
    for (size_t i = 0; i < 3; i++) {
      cin>> t;
      readFace(t);
    }
    cin>> t;
  }
  fclose (stdin);
  numNodes = indMap.size();
  std::cout << numNodes << '\n';

  std::cout << mx << ' ' << my << ' ' << mz << ' ' << '\n';
  std::cout << Mx << ' ' << My << ' ' << Mz << ' ' << '\n';
}

float distacia(int p1, int p2){
  float ac;
  ac = pow(gridMap[p1].x - gridMap[p2].x,2);
  ac += pow(gridMap[p1].y - gridMap[p2].y,2);
  ac += pow(gridMap[p1].z - gridMap[p2].z,2);
  return sqrt(ac);
}

void addPoint (int p1, int p2, int p3){
  float x, y, z;
  x = (gridMap[p1].x + gridMap[p2].x)/2;
  y = (gridMap[p1].y + gridMap[p2].y)/2;
  z = (gridMap[p1].z + gridMap[p2].z)/2;
  int p4 = gridMap.size();
  gridMap.push_back(point3( x, y, z));
  reducTri(p1, p4, p3);
  reducTri(p2, p4, p3);
}

void reducTri(int p1, int p2, int p3){
  float l[3], lm = 0;
  int i, j;
  l[0] = distacia(p1, p2);
  l[1] = distacia(p2, p3);
  l[2] = distacia(p3, p1);
  for (i = 0; i < 3; i++) {
    if (lm < l[i]) {
      lm = l[i];
      j = i;
    }
  }
  if(lm > R){
    if (j == 0) addPoint(p1, p2, p3);
    else if (j == 1) addPoint(p2, p3, p1);
    else addPoint(p3, p1, p2);
  }
}

void flood_fillXY(long long int x, long long int y, long long int z, int* voxInd,
long long int lx,long long int ly){
  long long int p = x + (y*lx) + (z*lx*ly);
  //printf("falla %lld\n",p);

  if (voxInd[p] > 0) return;
  voxInd[p] = 1;
  //gridMap.push_back(point3(x*R + mx, y*R + my, z*R + mz));
  flood_fillXY(x - 1, y , z, voxInd, lx, ly);
  flood_fillXY(x, y - 1, z, voxInd, lx, ly);
  flood_fillXY(x + 1, y , z, voxInd, lx, ly);
  flood_fillXY(x, y + 1, z, voxInd, lx, ly);
  //flood_fill(x, y , z - 1, voxInd, lx, ly);
  //flood_fill(x, y , z + 1, voxInd, lx, ly);
}
void flood_fillXZ(long long int x, long long int y, long long int z, int* voxInd,
long long int lx,long long int ly){
  long long int p = x + (y*lx) + (z*lx*ly);
  //printf("falla %lld\n",p);

  if (voxInd[p] > 1) return;
  voxInd[p] = 2;
  //gridMap.push_back(point3(x*R + mx, y*R + my, z*R + mz));
  flood_fillXZ(x - 1, y , z, voxInd, lx, ly);
  //flood_fill(x, y - 1, z, voxInd, lx, ly);
  flood_fillXZ(x, y , z - 1, voxInd, lx, ly);
  flood_fillXZ(x + 1, y , z, voxInd, lx, ly);
  //flood_fill(x, y + 1, z, voxInd, lx, ly);
  flood_fillXZ(x, y , z + 1, voxInd, lx, ly);
}
void flood_fillYZ(long long int x, long long int y, long long int z, int* voxInd,
long long int lx,long long int ly){
  long long int p = x + (y*lx) + (z*lx*ly);
  //printf("falla %lld\n",p);

  if (voxInd[p] > 2) return;
  voxInd[p] = 3;
  //gridMap.push_back(point3(x*R + mx, y*R + my, z*R + mz));
  //flood_fillYZ(x - 1, y , z, voxInd, lx, ly);
  flood_fillYZ(x, y - 1, z, voxInd, lx, ly);
  flood_fillYZ(x, y , z - 1, voxInd, lx, ly);
  //flood_fillYZ(x + 1, y , z, voxInd, lx, ly);
  flood_fillYZ(x, y + 1, z, voxInd, lx, ly);
  flood_fillYZ(x, y , z + 1, voxInd, lx, ly);
}
void flood_fill_XYZ(long long int x, long long int y, long long int z, int* voxInd,
long long int lx,long long int ly, long long int lz){
  long long int i;
  for (i = 1; i < lz - 1; i++) {
    flood_fillXY(x, y , i, voxInd, lx, ly);
  }
  for (i = 1; i < ly - 1; i++) {
    flood_fillXZ(x, i , z, voxInd, lx, ly);
  }
  for (i = 1; i < lx - 1; i++) {
    flood_fillXZ(i, y , z, voxInd, lx, ly);
  }
}

void newPoints(){
  long long int lx, ly, lz, i, x, y, z, p;
  lx = (long long int )((Mx - mx)/R) + 5;
  ly = (long long int )((My - my)/R) + 5;
  lz = (long long int )((Mz - mz)/R) + 5;
  int* voxInd = (int*) calloc(lx*ly*lz,sizeof(int));
  float* voxX = (float*) calloc(lx*ly*lz,sizeof(float));
  float* voxY = (float*) calloc(lx*ly*lz,sizeof(float));
  float* voxZ = (float*) calloc(lx*ly*lz,sizeof(float));
  for (i = 0; i < indMap.size(); i += 3) {
    reducTri(indMap[i], indMap[i + 1], indMap[i + 2]);
  }
  for (i = 0; i < gridMap.size(); i++) {
    x = (long long int )((gridMap[i].x - mx)/R) + 2;
    y = (long long int )((gridMap[i].y - my)/R) + 2;
    z = (long long int )((gridMap[i].z - mz)/R) + 2;
    p = x + (y*lx) + (z*lx*ly);
    voxInd[p]++;
    voxX[p] += gridMap[i].x;
    voxY[p] += gridMap[i].y;
    voxZ[p] += gridMap[i].z;
  }
  gridMap.clear();

  for (x = 0; x < lx; x++){
    for (y = 0; y < ly; y++){
      voxInd[x + (y*lx)] = 3;
      voxInd[x + (y*lx) + ((lz - 1)*lx*ly)] = 3;
    }
  }
  for (x = 0; x < lx; x++){
    for (z = 0; z < lz; z++){
      voxInd[x + (z*lx*ly)] = 3;
      voxInd[x + ((ly - 1)*lx) + (z*lx*ly)] = 3;
    }
  }
  for (z = 0; z < lz; z++){
    for (y = 0; y < ly; y++){
      voxInd[(y*lx) + (z*lx*ly)] = 3;
      voxInd[(lx - 1) + (y*lx) + (z*lx*ly)] = 3;
    }
  }
  //long long int a, ma = 10, xa, ya, ba, za, zz;

  for (x = 2; x < lx - 2; x++)
    for (y = 2; y < ly - 2; y++){
      for (z = 2; z < lz - 2; z++) {
        p = x + (y*lx) + (z*lx*ly);
        if (voxInd[p] > 0){
          gridMap.push_back(point3(voxX[p]/voxInd[p], voxY[p]/voxInd[p], voxZ[p]/voxInd[p]));
          gridMapColor.push_back(point3(1.0,0.0,0.0));
          voxInd[p] = 3;
        }
      }
    }

  //printf("falla 2, %lld - %lld - %lld \n", lx, ly, lz);
  flood_fill_XYZ(1, 1, 1, voxInd, lx, ly, lz);

  for (x = 2; x < lx - 2; x++)
    for (y = 2; y < ly - 2; y++){
      for (z = 2; z < lz - 2; z++) {
        p = x + (y*lx) + (z*lx*ly);
        if (voxInd[p] ==  0){
          gridMap.push_back(point3((x - 1.5)*R + mx, (y - 1.5)*R + my, (z -  1.5)*R + mz));
          gridMapColor.push_back(point3(1.0,1.0,1.0));
        }
      }
    }
}

ParticleSystem systemP = ParticleSystem();
tempSolver tp;
solverParams tempParams;

void NewSphere (){
  float angle = 90;
  for (size_t i = 0; i < gridMap.size(); i++) {
    vec4 p(Rotate(angle, -1, 0, 0) * vec4(gridMap[i], 1.0));
    tp.positionsControl.push_back(make_float4(p.x + 3, p.y + 0.1 - mz, p.z + 3, 1.0f));
  }
}


int XDIM = 128, YDIM = 128, ZDIM = 128;

void init(){

  cube_vertices.push_back( vec3(-1.0,  1.0,  1.0));
  cube_vertices.push_back( vec3(-1.0, -1.0,  1.0));
  cube_vertices.push_back( vec3( 1.0, -1.0,  1.0));
  cube_vertices.push_back( vec3( 1.0,  1.0,  1.0));
  cube_vertices.push_back( vec3(-1.0,  1.0, -1.0));
  cube_vertices.push_back( vec3(-1.0, -1.0, -1.0));
  cube_vertices.push_back( vec3( 1.0, -1.0, -1.0));
  cube_vertices.push_back( vec3( 1.0,  1.0, -1.0));

  readSphere("media/sphere.8");
  tp.spherePoints.resize(sphere_points.size());
  tempParams.numSpherePoints = sphere_points.size();
  for (size_t i = 0; i < sphere_points.size() ; i++) {
    tp.spherePoints[i].x = sphere_points[i].x;
    tp.spherePoints[i].y = sphere_points[i].y;
    tp.spherePoints[i].z = sphere_points[i].z;
  }
  readObje("media/dragon_smooth.obj");

  printf("falla 1 ---  \n");
  newPoints();
  NewSphere ();

  ControlBase scene("ControlBase");
	scene.init(&tp, &tempParams);
	systemP.initialize(tp, tempParams);

  glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_3D, textureID);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);


  glTexImage3D(GL_TEXTURE_3D,0,GL_RED, tempParams.gridWidth, tempParams.gridHeight, tempParams.gridDepth,0,GL_RED,GL_UNSIGNED_BYTE,NULL);
  glGenerateMipmap(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, 0);
  std::cout << "/* size grid */"<< tempParams.gridWidth*tempParams.gridHeight*tempParams.gridDepth << '\n';

  cudaGraphicsGLRegisterImage(&resources1, textureID, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

  glGenBuffers(1, &cube_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, cube_buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(point3)*36, NULL, GL_STATIC_DRAW);

  unsigned int sizeData = 0;

  for (int i = 0; i < 36; i++) {
    glBufferSubData(GL_ARRAY_BUFFER, sizeData, sizeof(point3), verticesRay[cubeIndicesRay[i]]);
    sizeData += sizeof(point3);
  }

  numNodes = tempParams.numParticles * tempParams.numSpherePoints;

  for (int i = 0; i < tempParams.numParticles; i++) {
    for (int j = 0; j < sphere_points.size(); j++) {
      spheres_norm_control.push_back(sphere_norm[j]);
      spheres_color_control.push_back(vec3(0.0, 1.0, 0.0));
    }
  }
  std::cout << tempParams.numParticles << '\n';

  glGenBuffers(1, &grid_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, grid_buffer);
	glBufferData(GL_ARRAY_BUFFER, numNodes * 3 * sizeof(float) , 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaGraphicsGLRegisterBuffer(&resources, grid_buffer, cudaGraphicsRegisterFlagsWriteDiscard);

  glGenBuffers(1, &grid_bufferAtrip);
  glBindBuffer(GL_ARRAY_BUFFER, grid_bufferAtrip);
  glBufferData(GL_ARRAY_BUFFER, sizeof(point3)*numNodes*2, NULL, GL_STATIC_DRAW);
  sizeData = 0;

  for (int i = 0; i < numNodes; i++) {
    glBufferSubData(GL_ARRAY_BUFFER, sizeData, sizeof(point3), spheres_norm_control[i]);


    sizeData += sizeof(point3);
  }


  for (int i = 0; i < numNodes; i++) {
    glBufferSubData(GL_ARRAY_BUFFER, sizeData, sizeof(point3),spheres_color_control[i]);
    sizeData += sizeof(point3);
  }

  program = InitShader("vshader3.glsl", "fshader3.glsl");

  glEnable( GL_DEPTH_TEST );
  glClearColor( 0.529, 0.807, 0.92, 0.0 );
  glLineWidth(2.0);
  systemP.running = true;
}

void drawObj(GLuint buffer, int num_vertices)
{
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  GLuint vPosition = glGetAttribLocation(program, "vPosition");
  glEnableVertexAttribArray(vPosition);
  glVertexAttribPointer(vPosition, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0) );

  glBindBuffer(GL_ARRAY_BUFFER, grid_bufferAtrip);
  GLuint vNormal = glGetAttribLocation(program, "vNormal");
  glEnableVertexAttribArray(vNormal);
  glVertexAttribPointer(vNormal, 3, GL_FLOAT, GL_FALSE, 0,
    BUFFER_OFFSET(0) );

  GLuint vColor = glGetAttribLocation(program, "vColor");
  glEnableVertexAttribArray(vColor);
  glVertexAttribPointer(vColor, 3, GL_FLOAT, GL_FALSE, 0,
    BUFFER_OFFSET(sizeof(point3) * num_vertices ) );

  glDrawArrays(GL_TRIANGLES, 0, num_vertices);
  glDisableVertexAttribArray(vPosition);
  glDisableVertexAttribArray(vNormal);
  glDisableVertexAttribArray(vColor);
}

void drawCube(GLuint buffer, int num_vertices)
{
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  GLuint vPosition = glGetAttribLocation(program, "vPosition");
  glEnableVertexAttribArray(vPosition);
  glVertexAttribPointer(vPosition, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0) );

  glDrawArrays(GL_TRIANGLES, 0, num_vertices);

  glDisableVertexAttribArray(vPosition);

}

void mainUpdate(ParticleSystem &system, tempSolver &tp, solverParams &tempParams) {

	system.updateWrapper(tempParams);

  dim3 textureDim(tempParams.gridWidth, tempParams.gridHeight, tempParams.gridDepth);

	void* positionsPtr;

	cudaCheck(cudaGraphicsMapResources(1, &resources1, NULL));
  cudaGraphicsSubResourceGetMappedArray(&cuda_image_array, resources1, 0, 0);
  system.getTexture(cuda_image_array, textureDim);

	cudaGraphicsUnmapResources(1, &resources1, NULL);

}



void display( void ){

  mainUpdate(systemP, tp, tempParams);
  glBindTexture(GL_TEXTURE_3D, textureID);

  GLuint  mv_matrix;
  GLuint  proj_matrix;
  GLuint  view_matrix;
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  glUseProgram(program);
  mv_matrix   = glGetUniformLocation(program, "MVP");
  view_matrix = glGetUniformLocation(program, "view_matrix");
  proj_matrix  = glGetUniformLocation(program, "proj_matrix");

  glUniform1i( glGetUniformLocation(program, "volume"), 0 );


  mat4  p = Perspective(fovy, aspect, zNear, zFar);

  glUniformMatrix4fv(proj_matrix, 1, GL_TRUE, p);
  mat4 m = Translate(eye) * rot;

  mat4  v = LookAt(VRP, VPN, VUP);
  vec3 camPos = (inverse(upperLeftMat3(rot)))*vec3(VRP[0] - eye[0],VRP[1] - eye[1],VRP[2] - eye[2]);

  glUniformMatrix4fv(mv_matrix, 1, GL_TRUE, p*v*m);
  glUniformMatrix4fv(view_matrix, 1, GL_TRUE, v);
  glUniform3f(glGetUniformLocation(program, "camPos"), camPos[0], camPos[1], camPos[2]);
  glUniform3f(glGetUniformLocation(program, "step_size"), 1.0f/XDIM, 1.0f/YDIM, 1.0f/ZDIM);

  if(flag_gird)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  else
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


  drawCube(cube_buffer, 36);


  glutSwapBuffers();
  glBindTexture(GL_TEXTURE_3D, 0);
}

void idle(void){
  glutPostRedisplay();
}

void reshape(int width, int height)
{
    glViewport(0, 0, width, height);
    aspect = (GLfloat) width  / (GLfloat) height;
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
  float a = 0.125;
  float angle = (10 / PI);
	switch (key) {
	case 033:
	case 'q': case 'Q':
		exit(EXIT_SUCCESS);
		break;

	case GLUT_KEY_UP: eye[0] += 1.0; break;
	case 'w': eye[0] += a; break;
	case 's': eye[0] -= a; break;
	case 'a': eye[2] += a; break;
	case 'd': eye[2] -= a; break;
	case 'Z': eye[1] -= a; break;
	case 'z': eye[1] += a; break;
	case '4': rot = Rotate(angle, 0, 1, 0)*rot;break;
	case '6': rot = Rotate(angle, 0, -1, 0)*rot;break;
	case '8': rot = Rotate(angle, 0, 0, 1)*rot;break;
	case '2': rot = Rotate(angle, 0, 0, -1)*rot;break;
  case 'g': flag_gird = 1 - flag_gird; break;

	case ' ':
		eye = vec4(0.0, 0.0, 0.0, 0.0);
    rot = Rotate(0, 1, 1, 1);
		break;
	}
	glutPostRedisplay();

}

int main(int argc, char **argv) {
  int err;

  cudaGLSetGLDevice(0);

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutCreateWindow("Practica04");
  err = glewInit();
  if (GLEW_OK != err)
  { printf("Error: glewInit failed: %s\n", (char*) glewGetErrorString(err));
    exit(1);
  }

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);
  glutIdleFunc(idle);

  init();
  glutMainLoop();
  return 0;
}
