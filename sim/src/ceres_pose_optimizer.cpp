// #include "ceres_pose_optimizer.hpp"
// #include <iostream>

// CeresPoseOptimizer::CeresPoseOptimizer(
//     const std::vector<Eigen::Vector3d>& ground_truth_points,
//     std::unordered_map<int, std::vector<int>> observed_points,
//     std::unordered_map<int, std::vector<Eigen::Vector3d>> backprojected_points,
//     std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& camera_poses)
//     : ground_truth_points_(ground_truth_points),
//       observed_points_(std::move(observed_points)),
//       backprojected_points_(std::move(backprojected_points)),
//       camera_poses_(camera_poses) {}

// void CeresPoseOptimizer::optimize() {
//     ceres::Problem problem;

//     for (const auto& [pose_index, observed_indices] : observed_points_) {
//         const auto& backprojected_pts = backprojected_points_[pose_index];

//         // Ensure we have a valid pose index
//         if (pose_index >= camera_poses_.size()) {
//             std::cerr << "Pose index " << pose_index << " is out of bounds." << std::endl;
//             continue;
//         }

//         // Convert SE3 pose to Eigen vector for optimization
//         Sophus::Vector6d pose_vector = camera_poses_[pose_index].log();

//         for (size_t i = 0; i < observed_indices.size(); ++i) {
//             int gt_index = observed_indices[i];
//             const Eigen::Vector3d& backprojected_point = backprojected_pts[i];
//             const Eigen::Vector3d& ground_truth_point = ground_truth_points_[gt_index];

//             ceres::CostFunction* cost_function = BackprojectionError::Create(backprojected_point, ground_truth_point);
//             ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
//             problem.AddResidualBlock(cost_function, loss_function, pose_vector.data());
//         }

//         // Update the camera pose with the optimized values
//         camera_poses_[pose_index] = Sophus::SE3d::exp(pose_vector);
//     }

//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_SCHUR;
//     options.minimizer_progress_to_stdout = true;
//     options.max_num_iterations = 200;
//     options.function_tolerance = 1e-9;  // Adjust as needed
//     options.gradient_tolerance = 1e-9;  // Adjust as needed
//     options.parameter_tolerance = 1e-9; // Adjust as needed

//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
//     std::cout << summary.FullReport() << std::endl;
// }

#include "ceres_pose_optimizer.hpp"
#include <iostream>

CeresPoseOptimizer::CeresPoseOptimizer(
    const std::vector<Eigen::Vector3d>& ground_truth_points,
    std::unordered_map<int, std::vector<int>> observed_points,
    std::unordered_map<int, std::vector<Eigen::Vector3d>> backprojected_points,
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& camera_poses)
    : ground_truth_points_(ground_truth_points),
      observed_points_(std::move(observed_points)),
      backprojected_points_(std::move(backprojected_points)),
      camera_poses_(camera_poses) {}

void CeresPoseOptimizer::optimize() {
    ceres::Problem problem;

    // Register each pose as a parameter block
    for (int pose_index = 0; pose_index < camera_poses_.size(); ++pose_index) {
        Sophus::SE3d& pose = camera_poses_[pose_index];
        problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters);  // Add pose parameter block to Ceres

        const auto& observed_indices = observed_points_[pose_index];
        const auto& backprojected_pts = backprojected_points_[pose_index];

        for (size_t i = 0; i < observed_indices.size(); ++i) {
            int gt_index = observed_indices[i];
            const Eigen::Vector3d& backprojected_point = backprojected_pts[i];
            const Eigen::Vector3d& ground_truth_point = ground_truth_points_[gt_index];

            ceres::CostFunction* cost_function = BackprojectionError::Create(backprojected_point, ground_truth_point);
            ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(cost_function, loss_function, pose.data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-9;  // Adjust as needed
    options.gradient_tolerance = 1e-9;  // Adjust as needed
    options.parameter_tolerance = 1e-9; // Adjust as needed

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
}
