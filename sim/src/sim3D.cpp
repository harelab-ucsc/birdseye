#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip> 
#include <yaml-cpp/yaml.h>
#include <limits>
#include <algorithm>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include "ortho_synth.hpp"

typedef Eigen::Vector3d Point;
typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> Trajectory;

// Function to generate random 3D ground points within a circular boundary
std::vector<Point> generateRandomPoints(int n, double boundary_radius, double height_max) {
    std::vector<Point> points;
    for (int i = 0; i < n; ++i) {
        double r = boundary_radius * sqrt(static_cast<double>(rand()) / RAND_MAX);
        double theta = 2 * M_PI * static_cast<double>(rand()) / RAND_MAX;
        double height = height_max * static_cast<double>(rand()) / RAND_MAX;
        Point point(r * cos(theta), r * sin(theta), height);
        points.push_back(point);
    }
    return points;
}

// Function to calculate Euclidean distance between two points
double distance(const Point& p1, const Point& p2) {
    return sqrt(pow(p1.x() - p2.x(), 2) + pow(p1.y() - p2.y(), 2));
}

// Function to perform simple clustering based on a given radius
std::vector<Point> clusterPoints(const std::vector<Point>& points, double cluster_radius) {
    std::vector<Point> centroids;
    std::vector<bool> visited(points.size(), false);

    for (size_t i = 0; i < points.size(); ++i) {
        if (!visited[i]) {
            Point centroid(0.0, 0.0, 0.0);
            int count = 0;
            for (size_t j = 0; j < points.size(); ++j) {
                if (distance(points[i], points[j]) <= cluster_radius) {
                    centroid.x() += points[j].x();
                    centroid.y() += points[j].y();
                    visited[j] = true;
                    count++;
                }
            }
            centroid.x() /= count;
            centroid.y() /= count;
            centroids.push_back(centroid);
        }
    }
    return centroids;
}

// Function to generate a TSP path using a naive nearest-neighbor approach
std::vector<int> solveTSP(const std::vector<Point>& centroids) {
    std::vector<int> tsp_path;
    std::vector<bool> visited(centroids.size(), false);
    int current = 0;
    tsp_path.push_back(current);
    visited[current] = true;

    for (size_t i = 1; i < centroids.size(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int next = -1;

        for (size_t j = 0; j < centroids.size(); ++j) {
            if (!visited[j] && distance(centroids[current], centroids[j]) < min_dist) {
                min_dist = distance(centroids[current], centroids[j]);
                next = j;
            }
        }
        current = next;
        tsp_path.push_back(current);
        visited[current] = true;
    }
    return tsp_path;
}

// Function to generate trajectory points based on TSP path
Trajectory generateTrajectory(const std::vector<Point>& centroids, const std::vector<int>& tsp_path, int m, double altitude) {
    Trajectory trajectory;

    for (size_t i = 0; i < tsp_path.size(); ++i) {
        const Point start = centroids[tsp_path[i]];
        const Point end = centroids[tsp_path[(i + 1) % tsp_path.size()]];

        for (int j = 0; j < m / tsp_path.size(); ++j) {
            double alpha = static_cast<double>(j) / (m / tsp_path.size() - 1);

            // Interpolated position
            Eigen::Vector3d position(
                start.x() * (1 - alpha) + end.x() * alpha,
                start.y() * (1 - alpha) + end.y() * alpha,
                altitude
            );

            // Direction vector from start to end
            Eigen::Vector3d direction = (Eigen::Vector3d(end.x(), end.y(), altitude) -
                                         Eigen::Vector3d(start.x(), start.y(), altitude)).normalized();

            // z-axis points down
            Eigen::Vector3d z_axis(0.0, 0.0, -1.0);

            // x-axis points in the direction of travel
            Eigen::Vector3d x_axis = direction.normalized();

            // y-axis is orthogonal to x and z (right-hand rule)
            Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();

            // Construct rotation matrix
            Eigen::Matrix3d rotation;
            rotation.col(0) = x_axis;
            rotation.col(1) = y_axis;
            rotation.col(2) = z_axis;

            // Create SE3 element
            Sophus::SE3d pose(rotation, position);

            // Add to trajectory
            trajectory.push_back(pose);
        }
    }

    return trajectory;
}

// Function to add zero-mean Gaussian noise to the trajectory and return a new vector with noisy poses
Trajectory addGaussianNoiseToTrajectory(const Trajectory &trajectory, 
                             const Eigen::Vector3d& translation_stddev, 
                             const Eigen::Vector3d& rotation_stddev) {
    // Create random number generators
    std::default_random_engine generator;

    // Create individual normal distributions for each translation component
    std::normal_distribution<double> translation_dist_x(0.0, translation_stddev.x());
    std::normal_distribution<double> translation_dist_y(0.0, translation_stddev.y());
    std::normal_distribution<double> translation_dist_z(0.0, translation_stddev.z());

    // Create individual normal distributions for each rotation component
    std::normal_distribution<double> rotation_dist_x(0.0, rotation_stddev.x());
    std::normal_distribution<double> rotation_dist_y(0.0, rotation_stddev.y());
    std::normal_distribution<double> rotation_dist_z(0.0, rotation_stddev.z());

    // Create a new vector to store the noisy poses
    Trajectory noisy_trajectory;
    noisy_trajectory.reserve(trajectory.size());

    for (const auto& pose : trajectory) {
        // Add noise to the translation component
        Eigen::Vector3d translation_noise(
            translation_dist_x(generator),
            translation_dist_y(generator),
            translation_dist_z(generator)
        );
        Eigen::Vector3d noisy_translation = pose.translation() + translation_noise;

        // Add noise to the rotation component
        Eigen::Vector3d rotation_noise(
            rotation_dist_x(generator),
            rotation_dist_y(generator),
            rotation_dist_z(generator)
        );
        Sophus::SO3d rotation_noise_SO3 = Sophus::SO3d::exp(rotation_noise);
        Sophus::SO3d noisy_rotation = rotation_noise_SO3 * pose.so3();

        // Create the noisy SE3 pose
        Sophus::SE3d noisy_pose(noisy_rotation, noisy_translation);
        noisy_trajectory.push_back(noisy_pose);
    }

    return noisy_trajectory;
}

// Function to add Gaussian noise to pixel coordinates
Eigen::Vector2d addGaussianPixelError(const Eigen::Vector2d& pixel, double stddev_x, double stddev_y) {
    // Create a random number generator and normal distributions for x and y
    std::default_random_engine generator;
    std::normal_distribution<double> dist_x(0.0, stddev_x);
    std::normal_distribution<double> dist_y(0.0, stddev_y);

    // Generate Gaussian noise
    double noise_x = dist_x(generator);
    double noise_y = dist_y(generator);

    // Add the noise to the original pixel coordinates
    Eigen::Vector2d noisy_pixel = pixel + Eigen::Vector2d(noise_x, noise_y);

    return noisy_pixel;
}

// Converts a 2D pixel coordinate to a 3D point in camera space
Eigen::Vector3d pixelTo3D(const Eigen::Vector2d &pixel, double depth, const Eigen::Matrix3d &K) {
    // Camera intrinsic matrix (K):
    // K = [ fx  0  cx ]
    //     [  0 fy  cy ]
    //     [  0  0   1 ]
    
    // Inverse of the intrinsic matrix
    Eigen::Matrix3d K_inv = K.inverse();

    // Convert pixel to normalized camera coordinates
    Eigen::Vector3d homogenous_pixel(pixel.x(), pixel.y(), 1.0);
    Eigen::Vector3d normalized_point = K_inv * homogenous_pixel;

    // Scale by depth to get 3D coordinates in the camera frame
    Eigen::Vector3d point_3d = normalized_point * depth;

    return point_3d;
}

// Projects a 3D point in camera space to a 2D pixel coordinate
Eigen::Vector2d space3DToPixel(const Eigen::Vector3d &point_3d, const Eigen::Matrix3d &K) {
    // Camera intrinsic matrix (K):
    // K = [ fx  0  cx ]
    //     [  0 fy  cy ]
    //     [  0  0   1 ]

    // Project the 3D point into the image plane
    Eigen::Vector3d projected_point = K * point_3d;

    // Normalize the homogeneous coordinates
    Eigen::Vector2d pixel(projected_point.x() / projected_point.z(), 
                          projected_point.y() / projected_point.z());

    return pixel;
}

// produce projected pixel 
// void backproject(const std::vector<Point> &ground_points, const Trajectory &trajectory_gt, 
//                  const Trajectory &trajectory_noisy, std::map<int, std::vector<Eigen::Vector3d>> &backprojections, 
//                  std::vector<double> &backproject_error, const Eigen::Matrix3d &K, 
//                  const Eigen::Vector2d &pixelRPE, int px_width, int px_height) {
//     // for each camera pose
//     for (int i = 0; i < trajectory_gt.size(); i++) {
//         // for each ground point w.r.t each camera pose 
//         for (int j = 0; j < ground_points.size(); j ++) {
//             Point P_camera = trajectory_gt[i].inverse() * ground_points[j]; // transform world observation to camera frame 3D coordinate
//             Eigen::Vector2d pixel = space3DToPixel(P_camera, K); // transform camera frame observation to pixel space 
//             Eigen::Vector2d pixel_noisy = addGaussianPixelError(pixel, pixelRPE.x(), pixelRPE.y());
//             // if pixel is within view of the camera frame 
//              if (pixel_noisy[0] >= 0 && pixel_noisy[0] < px_width &&
//                  pixel_noisy[1] >= 0 && pixel_noisy[0] < px_height) {
//                     Point loc = trajectory_gt[i].translation();
//                     double depth = loc.z(); // !!!image-wise depth assumption from z-component of pose estimate!!!
//                     Eigen::Vector3d P_camera_b = pixelTo3D(pixel, depth, K); // backproject pixel to camera frame
//                     Eigen::Vector3d P_world_b = trajectory_noisy[i].rotationMatrix() * P_camera_b + trajectory_noisy[i].translation(); // backproject to world frame
//                     backprojections.push_back(P_world_b); // save world pixel backprojection
//                     double backprojection_e = (ground_points[j] - P_world_b).norm(); // calculate L2 distance between g.t. observtaion and backprojected world coordinate
//                     backproject_error.push_back(backprojection_e);
//             }
//         }
//     }
// }

void backproject(const std::vector<Point> &ground_points, const Trajectory &trajectory_gt, 
                 const Trajectory &trajectory_noisy, std::map<int, std::vector<Eigen::Vector3d>> &backprojections,
                 const Eigen::Matrix3d &K, const Eigen::Vector2d &pixelRPE, int px_width, int px_height) {
    // for ground observation
    for (int i = 0; i < ground_points.size(); i++) {
        std::vector<Eigen::Vector3d> b_projs;
        // for each image pose
        for (int j = 0; j < trajectory_gt.size(); j++) {
            Point P_camera = trajectory_gt[j].inverse() * ground_points[i]; // transform world observation to camera frame 3D coordinate
            Eigen::Vector2d pixel = space3DToPixel(P_camera, K); // transform camera frame observation to pixel space 
            Eigen::Vector2d pixel_noisy = addGaussianPixelError(pixel, pixelRPE.x(), pixelRPE.y());
            // if pixel is within view of the camera frame 
             if (pixel_noisy[0] >= 0 && pixel_noisy[0] < px_width &&
                 pixel_noisy[1] >= 0 && pixel_noisy[0] < px_height) {
                    Point loc = trajectory_gt[j].translation();
                    double depth = loc.z(); // !!!image-wise depth assumption from z-component of pose estimate!!!
                    Eigen::Vector3d P_camera_b = pixelTo3D(pixel, depth, K); // backproject pixel to camera frame
                    Eigen::Vector3d P_world_b = trajectory_noisy[j].rotationMatrix() * P_camera_b + trajectory_noisy[j].translation(); // backproject to world frame
                    //backprojections.push_back(P_world_b); // save world pixel backprojection
                    b_projs.push_back(P_world_b);
                    //double backprojection_e = (ground_points[i] - P_world_b).norm(); // calculate L2 distance between g.t. observtaion and backprojected world coordinate
                    //backproject_error.push_back(backprojection_e);
            }
        }
        backprojections[i] = b_projs;
    }
}

// calculate l2 error between g.t. and backprojections
std::vector<double> get_l2_errors(const std::map<int, std::vector<Eigen::Vector3d>> &backprojections, const std::vector<Point> &ground_points) {
    std::vector<double> l2_errors;
    for (const auto& [index, estimations] : backprojections) {
        for (const auto& est : estimations) {
            double error = (ground_points[index] - est).norm(); // calculate L2 distance between g.t. observtaion and backprojected world coordinate
            l2_errors.push_back(error);
        }
    }
    return l2_errors;
}

// write errors to text file
void writeVectorToFile(const std::vector<double> &vec, const std::string &filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    
    for (const double& value : vec) {
        file << std::fixed << std::setprecision(6) << value << std::endl; // Write each value to a new line with precision
    }
    
    file.close();
    std::cout << "[]Results successfully written to " << filename << std::endl;
}

// Function to visualize the points, SE3 elements, and boundary using Pangolin
void visualize(const std::vector<Point> &points, 
               std::map<int, std::vector<Eigen::Vector3d>> &backprojections, 
               const Trajectory &trajectory,
               double boundary_radius) {
    pangolin::CreateWindowAndBind("Trajectory Visualization", 640, 480);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Calculate a good starting point for the camera's distance
    double cam_distance = boundary_radius * 3.0;

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(
            640, 480, 500, 500, 320, 389, 0.1, 1000
        ),
        pangolin::ModelViewLookAt(
            cam_distance, cam_distance, cam_distance,  // Camera position
            0, 0, 0,                                  // Look at point
            pangolin::AxisZ                           // Up direction (Z-axis)
        )
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(150), 1.0, -640.0f / 480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        // Set background color to white
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        // Draw the circular boundary in dark grey
        glColor3f(0.2f, 0.2f, 0.2f);  // Dark grey
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i <= 360; ++i) {
            double theta = i * M_PI / 180;
            glVertex3f(boundary_radius * cos(theta), boundary_radius * sin(theta), 0.0);
        }
        glEnd();

        // Draw the random ground points in green
        glColor3f(0.0f, 1.0f, 0.0f);  // Green
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (const auto& p : points) {
            glVertex3f(p.x(), p.y(), p.z());
        }
        glEnd();

        // Draw the b_proj points in blue
        glColor3f(0.0f, 0.0f, 1.0f);  // Blue
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (const auto& [index, estimations] : backprojections) {
            for (const auto& est : estimations) {
                glVertex3f(est.x(), est.y(), est.z());
            }
        }
        glEnd();

        // Draw the trajectory SE3 elements as RGB triads
        for (const auto& pose : trajectory) {
            Eigen::Matrix3d rotation = pose.rotationMatrix();
            Eigen::Vector3d translation = pose.translation();

            // X-axis in red
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(translation.x(), translation.y(), translation.z());
            Eigen::Vector3d x_axis_end = translation + 1.0 * rotation.col(0);
            glVertex3f(x_axis_end.x(), x_axis_end.y(), x_axis_end.z());
            glEnd();

            // Y-axis in green
            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(translation.x(), translation.y(), translation.z());
            Eigen::Vector3d y_axis_end = translation + 1.0 * rotation.col(1);
            glVertex3f(y_axis_end.x(), y_axis_end.y(), y_axis_end.z());
            glEnd();

            // Z-axis in blue
            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            glVertex3f(translation.x(), translation.y(), translation.z());
            Eigen::Vector3d z_axis_end = translation + 1.0 * rotation.col(2);
            glVertex3f(z_axis_end.x(), z_axis_end.y(), z_axis_end.z());
            glEnd();
        }

        pangolin::FinishFrame();
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path to parameter yaml>" << std::endl;
        return 1;
    }

    // Load the YAML file 
    std::string yaml_file = argv[1];
    YAML::Node param;
    try {
        param = YAML::LoadFile(yaml_file);
    } catch (const std::exception &e) {
        std::cerr << "Error loading parameter YAML file: " << e.what() << std::endl;
        return 1;
    }

    // load simulation parameters
    int n = param["num_points"].as<int>(); // Number of random ground observations
    double boundary_radius = param["boundary_radius"].as<double>(); // Radius of the circular boundary
    double ground_height_max = param["ground_height_max"].as<double>(); // max height of random ground points (0.0 is global flat-Earth assumption)
    double cluster_radius = param["cluster_radius"].as<double>();  // Radius for clustering 
    int m = param["num_poses"].as<int>();  // Number of trajectory points
    double flight_AGL = param["flight_AGL"].as<double>();  // flight altitude in meters 

    // load camera intrinsics
    int px_width = param["image_width"].as<int>();
    int px_height = param["image_width"].as<int>();
    double fx = param["fx"].as<double>();
    double fy = param["fy"].as<double>();
    double cx = param["cx"].as<double>();
    double cy = param["cy"].as<double>();
    // create camera matrix K
    Eigen::Matrix3d K;
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;

    // load noise params
    // std for (x, y, z) [m] pose error
    double sigma_x = param["sigma_x"].as<double>();
    double sigma_y = param["sigma_y"].as<double>();
    double sigma_z = param["sigma_z"].as<double>();

    // std for (roll, pitch, yaw) [rad] pose error
    double sigma_roll = param["sigma_roll"].as<double>();
    double sigma_pitch = param["sigma_pitch"].as<double>();
    double sigma_yaw = param["sigma_yaw"].as<double>();

    // pixel reprojection error (perspective and lens distortion calibration error)
    double sigma_RPE_x = param["sigma_RPE_x"].as<double>();
    double sigma_RPE_y = param["sigma_RPE_y"].as<double>();

    // load ortho-synthetic filtering params
    double eps = param["eps"].as<double>();
    int min_points = param["min_points"].as<int>();

    Eigen::Vector3d translation_stddev(sigma_x, sigma_y, sigma_z);
    Eigen::Vector3d rotation_stddev(sigma_roll, sigma_pitch, sigma_yaw);
    Eigen::Vector2d pixelRPE(sigma_RPE_x, sigma_RPE_y);

    std::cout << "<><><><><><><><><>" << std::endl;
    std::cout << "BirdsEye Simulator" << std::endl;
    std::cout << "<><><><><><><><><>" << std::endl;
    std::cout << "Simulation parameters:" << std::endl;
    std::cout << "\tNumber of ground observations = " << n << std::endl;
    double test_area = pow(boundary_radius, 2.0) * M_PI;
    std::cout << "\tTesting area = " << test_area << " m^2 (" << test_area / 4047 << " acres)" << std::endl;
    std::cout << "\tMax ground height = " << ground_height_max << " m" << std::endl;
    std::cout << "\tCluster radius = " << cluster_radius << " m" << std::endl;
    std::cout << "\tNumber of images = " << m << std::endl;
    std::cout << "\tFlight Altitude (AGL) = " << flight_AGL << " m" << std::endl;
    std::cout << std::endl << "Measurement uncertainties:" << std::endl;
    std::cout << "\tTranslation estimation std [x, y, z] =        [" << sigma_x << ", " << sigma_y << ", " << sigma_z << "]" << std::endl;
    std::cout << "\tAttitude estimation std [roll, pitch, yaw] =  [" << sigma_roll << ", " << sigma_pitch << ", " << sigma_yaw << "]" << std::endl;
    std::cout << "\tPixel reprojection error std [RPE_x, RPE_y] = [" << sigma_RPE_x << ", " << sigma_RPE_y << "]" << std::endl;

    std::cout << std::endl << "Starting simulation" << std::endl;
    // Generate random ground observations
    std::cout << "[]Generating " << n << " random ground observations..." << std::endl;
    srand(static_cast<unsigned int>(time(0))); // Seed the random number generator with the current time
    std::vector<Point> points = generateRandomPoints(n, boundary_radius, ground_height_max);

    // Cluster the points and get centroids
    std::cout << "[]Clustering ground points to generate flight waypoints..." << std::endl;
    std::vector<Point> centroids = clusterPoints(points, cluster_radius);

    // Solve TSP to get the order of centroids
    std::cout << "[]Generating optimal flight path..." << std::endl;
    std::vector<int> tsp_path = solveTSP(centroids);

    // Generate the ground truth trajectory based on TSP path
    Trajectory trajectory_gt = generateTrajectory(centroids, tsp_path, m, flight_AGL);
    std::cout << "[]Simulating noisy flight trajectory..." << std::endl;
    Trajectory trajectory_noisy = addGaussianNoiseToTrajectory(trajectory_gt, translation_stddev, rotation_stddev);

    // backprojection results
    std::cout << "[]Backprojecting points from pixel space to world space..." << std::endl;
    std::map<int, std::vector<Eigen::Vector3d>> backprojections; // dictionary (ground point index -> set of corresponding backprojections)
    backproject(points, trajectory_gt, trajectory_noisy, backprojections, K, pixelRPE, px_width, px_height); // perform backprojection
    std::vector<double> backproject_error_raw = get_l2_errors(backprojections, points);
    std::cout << "[]Simulated " << backproject_error_raw.size() << " backprojections." << std::endl;

    // Write the error terms to the file
    writeVectorToFile(backproject_error_raw, "l2_errors_raw.txt");

    // ortho-synthetic filtering
    std::cout << "[]Performing ortho-synthetic filtering..." << std::endl;
    ortho_synth::Filter filtering(eps, min_points);
    // Perform DBSCAN and get filtered results
    std::map<int, std::vector<Eigen::Vector3d>> filtered_backprojections = filtering.removeOutliers(backprojections);
    std::vector<double> backproject_error_filtered = get_l2_errors(filtered_backprojections, points);
    // Write the error terms to the file
    writeVectorToFile(backproject_error_filtered, "l2_errors_filtered.txt");

    // Visualize the points, trajectory, and boundary
    visualize(points, filtered_backprojections, trajectory_noisy, boundary_radius);

    return 0;
}