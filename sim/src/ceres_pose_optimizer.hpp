#ifndef CERES_POSE_OPTIMIZER_HPP
#define CERES_POSE_OPTIMIZER_HPP

#include <unordered_map>
#include <vector>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>

class CeresPoseOptimizer {
public:
    CeresPoseOptimizer(
        const std::vector<Eigen::Vector3d>& ground_truth_points,
        std::unordered_map<int, std::vector<int>> observed_points,
        std::unordered_map<int, std::vector<Eigen::Vector3d>> backprojected_points,
        std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& camera_poses);

    void optimize();

private:
    struct BackprojectionError {
        BackprojectionError(const Eigen::Vector3d backprojected_point, const Eigen::Vector3d& ground_truth_point)
            : backprojected_point_(backprojected_point), ground_truth_point_(ground_truth_point) {}

        template <typename T>
        bool operator()(const T* const pose_data, T* residual) const {
            // Convert pose_data to an SE3 object using the exponential map
            Sophus::SE3<T> se3_pose = Sophus::SE3<T>::exp(Eigen::Map<const Sophus::Vector6<T>>(pose_data));

            // Convert points to type T
            Eigen::Matrix<T, 3, 1> backprojected_point_T = backprojected_point_.cast<T>();
            Eigen::Matrix<T, 3, 1> ground_truth_point_T = ground_truth_point_.cast<T>();

            // Transform the backprojected point from the camera frame to the world frame
            Eigen::Matrix<T, 3, 1> transformed_point = se3_pose * backprojected_point_T;

            // Compute the residuals as the difference between the transformed backprojected point and the ground truth point
            residual[0] = transformed_point.x() - ground_truth_point_T.x();
            residual[1] = transformed_point.y() - ground_truth_point_T.y();
            residual[2] = transformed_point.z() - ground_truth_point_T.z();
            
            return true;
        }

        static ceres::CostFunction* Create(const Eigen::Vector3d backprojected_point, const Eigen::Vector3d& ground_truth_point) {
            return new ceres::AutoDiffCostFunction<BackprojectionError, 3, Sophus::SE3d::num_parameters>(
                new BackprojectionError(backprojected_point, ground_truth_point));
        }

        Eigen::Vector3d backprojected_point_;
        Eigen::Vector3d ground_truth_point_;
    };


    const std::vector<Eigen::Vector3d>& ground_truth_points_;
    std::unordered_map<int, std::vector<int>> observed_points_;
    std::unordered_map<int, std::vector<Eigen::Vector3d>> backprojected_points_;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& camera_poses_; // Reference to camera poses that will be optimized
};

#endif // CERES_POSE_OPTIMIZER_HPP
