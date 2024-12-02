#include "ortho_synth.hpp"
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>

namespace ortho_synth {

Filter::Filter(double eps, int minPts) : eps_(eps), minPts_(minPts) {}

std::map<int, std::vector<Eigen::Vector3d>> Filter::removeOutliers(const std::map<int, std::vector<Eigen::Vector3d>> &backprojections) const {
    std::map<int, std::vector<Eigen::Vector3d>> filtered_backprojections;

    // Iterate over each point cloud in the map
    for (const auto &pair : backprojections) {
        int key = pair.first;
        const std::vector<Eigen::Vector3d> &points = pair.second;

        // Convert input points to PCL format
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &p : points) {
            cloud->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
        }

        // Create the KdTree object for the search method of the extraction
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud);

        // Vector to store cluster indices
        std::vector<pcl::PointIndices> cluster_indices;

        // Perform DBSCAN clustering
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(eps_); // Distance tolerance
        ec.setMinClusterSize(minPts_); // Minimum number of points in a cluster
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        // Extract points that are part of clusters (not outliers)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        for (const auto &indices : cluster_indices) {
            inliers->indices.insert(inliers->indices.end(), indices.indices.begin(), indices.indices.end());
        }
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false); // Extract inliers
        extract.filter(*cloud_filtered);

        // Convert back to std::vector<Eigen::Vector3d>
        std::vector<Eigen::Vector3d> filtered_points;
        for (const auto &p : cloud_filtered->points) {
            filtered_points.emplace_back(p.x, p.y, p.z);
        }

        // Store filtered points in the new map
        filtered_backprojections[key] = filtered_points;
    }

    return filtered_backprojections;
}

}
