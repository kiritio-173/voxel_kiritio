#ifndef _COMMON_FUNCTIONS_
#define _COMMON_FUNCTIONS_
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Core>
#include <cstdio>

#include "nanoflann.hpp"
#include "utils.h"

using namespace cv;
using namespace std;
using namespace nanoflann;

class LineFunctions
{
public:
	LineFunctions(void){};
	~LineFunctions(void){};

public:
	static void lineFitting( int rows, int cols, std::vector<cv::Point> &contour, double thMinimalLineLength, std::vector<std::vector<cv::Point2d> > &lines );

	static void subDivision( std::vector<std::vector<cv::Point> > &straightString, std::vector<cv::Point> &contour, int first_index, int last_index
		, double min_deviation, int min_size  );

	static void lineFittingSVD( cv::Point *points, int length, std::vector<double> &parameters, double &maxDev );
};

struct PCAInfo
{
	double lambda0, scale;
	cv::Matx31d normal, planePt;
	std::vector<int> idxAll, idxIn;

	PCAInfo &operator =(const PCAInfo &info)
	{
		this->lambda0 = info.lambda0;
		this->normal = info.normal;
		this->idxIn = info.idxIn;
		this->idxAll = info.idxAll;
		this->scale = scale;
		return *this;
	}
};

class PCAFunctions 
{
public:
	PCAFunctions(void){};
	~PCAFunctions(void){};

	void Ori_PCA( PointCloud<double> &cloud, int k, std::vector<PCAInfo> &pcaInfos, double &scale, double &magnitd );

	void PCASingle( std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo );

	void MCMD_OutlierRemoval( std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo );

	double meadian( std::vector<double> dataset );
};

/************************************************************************/
// 体素相关结构体
class VOXEL_LOC 
{
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}
  
  VOXEL_LOC(const VOXEL_LOC &info){
    x = info.x;
    y = info.y;
    z = info.z;
  }

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <> struct hash<VOXEL_LOC> {
  size_t operator()(const VOXEL_LOC &s) const {
    using std::size_t;
    using std::hash;
    return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^
           (hash<int64_t>()(s.z) << 1);
  }
};
}

typedef struct Voxel {
  float size;
  Eigen::Vector3d voxel_origin;
  Eigen::Vector3d voxel_color;
  PointCloud<double> *cloud;
  VOXEL_LOC LOC;
  // 保存体素的法向量
  cv::Matx31d normal;
  double projDist;
  Voxel(){};
  Voxel(float _size) : size(_size) {
    voxel_origin << 0, 0, 0;
    cloud = new PointCloud<double>;
  };

  bool operator<(const Voxel &other) const{
    return LOC.x<other.LOC.x&&LOC.y<other.LOC.y&&LOC.z<other.LOC.z;
  }


  Voxel (const Voxel &other){
    cloud = new PointCloud<double>;
    cloud->pts = other.cloud->pts;
    size = other.size;
    normal = other.normal;
    projDist = other.projDist;
    LOC = other.LOC;
    voxel_origin = other.voxel_origin;
    voxel_color = other.voxel_color;
  }
  ~Voxel(){
    //释放构造函数中new的内存空间
    delete cloud;
  }
} Voxel;

typedef struct VoxelGrid {
  float size = 0.5;
  int index;
  Eigen::Vector3d origin;
  PointCloud<double> cloud;
} VoxelGrid;

struct M_POINT {
  float xyz[3];
  float intensity;
  int count = 0;
};

void initVoxel(
    const PointCloud<double> &input_cloud,
    const float voxel_size, std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map); 

void copyPointCloud(PointCloud<double> &cloud, PointCloud<double> &copy_container);

void turnFormat(PointCloud<double> &cloud, std::vector<std::vector<double>> &container);

void voxelNormal(const PCAInfo &pcaInfo, Voxel &v);

void calNormProj(const PCAInfo &pcaInfo, Voxel &v );


/*******************************************************************************/
#endif //_COMMON_FUNCTIONS_
