#ifndef ORTHO_SYNTH_H
#define ORTHO_SYNTH_H

#include <vector>
#include <map>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace ortho_synth {

class Filter {
public:
    // Constructor
    Filter(double eps, int minPts);

    // Method to remove outliers
    std::map<int, std::vector<Eigen::Vector3d>> removeOutliers(const std::map<int, std::vector<Eigen::Vector3d>> &backprojections) const;
    
private:
    double eps_;   // Maximum distance between two points to be considered neighbors
    int minPts_;   // Minimum number of points required to form a cluster
};

}

#endif 
