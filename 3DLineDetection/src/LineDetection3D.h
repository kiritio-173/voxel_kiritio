#ifndef _LINE_DETECTION_H_
#define _LINE_DETECTION_H_
#pragma once

#include "CommonFunctions.h"
#include <ctime>
#include <cstdlib>

struct PLANE
{
	double scale;
	std::vector<std::vector<std::vector<cv::Point3d> > > lines3d;

	PLANE &operator =(const PLANE &info)
	{
		this->scale    = info.scale;
		this->lines3d     = info.lines3d;
		return *this;
	}
};

class LineDetection3D 
{
public:
	LineDetection3D();
	~LineDetection3D();

	void run( PointCloud<double> &data, int k, std::vector<std::vector<int> > &regions, std::vector<PLANE> &planes, std::vector<std::vector<cv::Point3d> > &lines, std::vector<double> &ts );

	void pointCloudSegmentation( PointCloud<double> &pointData, std::vector<std::vector<int> > &regions );
	//修改版
	void pointCloudSegmentation_ver2( std::vector<std::vector<double> > &rpointData, Voxel *v);

	void planeBased3DLineDetection( std::vector<std::vector<int> > &regions, std::vector<PLANE> &planes );

	void postProcessing( std::vector<PLANE> &planes, std::vector<std::vector<cv::Point3d> > &lines );

	void voxelFiltrate(Voxel *v ,std::vector<Voxel> &voxel_tar,double degThre);

	int getRandomSeedIndex(const std::vector<Voxel>& voxels);

	bool isInvector(const std::vector<VOXEL_LOC> &voxel_locs,const VOXEL_LOC neighbor);

	void voxelGrow(std::vector<Voxel > *voxel_map ,std::vector<VOXEL_LOC> &voxelseed,int seedindex);
	// 
	void regionGrow( double thAngle, std::vector<std::vector<int> > &regions );

	void regionMerging( double thAngle, std::vector<std::vector<int> > &regions );

	void exportRegionCloud( const std::vector<std::vector<int> > regions ,std::vector<PCAInfo> patches,double degThre );
	void exportRegionCloud( const std::vector<std::vector<int> > regions );

	bool maskFromPoint( std::vector<cv::Point2d> &pts2d, double radius, double &xmin, double &ymin, double &xmax, double &ymax, int &margin, cv::Mat &mask );

	void lineFromMask( cv::Mat &mask, int thLineLengthPixel, std::vector<std::vector<std::vector<cv::Point2d> > > &lines );

	void outliersRemoval( std::vector<PLANE> &planes );

	void lineMerging( std::vector<PLANE> &planes, std::vector<std::vector<cv::Point3d> > &lines );
	
	void exportVoxelCloud(Voxel *v ,double degThre, const char* path );

public:
	int k;
	int pointNum;
	double scale, magnitd;
	std::vector<PCAInfo> pcaInfos;
	PointCloud<double> pointData;
};

#endif //_LINE_DETECTION_H_
