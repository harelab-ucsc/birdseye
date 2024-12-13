#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pangolin/pangolin.h>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <filesystem>

typedef Eigen::Vector3d Point;

// Function to load poses from a JSON file
std::map<std::string, Eigen::Affine3d> loadPosesFromJson(const std::string &jsonFile) {
    std::map<std::string, Eigen::Affine3d> poses;
    std::ifstream file(jsonFile);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open JSON file: " + jsonFile);
    }

    nlohmann::json jsonData;
    file >> jsonData;

    for (const auto &frame : jsonData["frames"]) {
        std::string imagePath = frame["file_path"];
        Eigen::Matrix4d transformMatrix;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                transformMatrix(i, j) = frame["transform_matrix"][i][j];
            }
        }
        poses[imagePath] = Eigen::Affine3d(transformMatrix);
    }

    return poses;
}

// Function to detect AprilTags in an image and filter for fully visible tags
std::vector<std::vector<cv::Point2f>> detectFullyVisibleAprilTags(const cv::Mat &image, 
                                                                  const cv::Ptr<cv::aruco::Dictionary> &arucoDict) {
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(image, arucoDict, markerCorners, markerIds);

    // Filter markers for full visibility
    std::vector<std::vector<cv::Point2f>> fullyVisibleMarkers;
    for (const auto &corners : markerCorners) {
        bool isFullyVisible = true;
        for (const auto &corner : corners) {
            if (corner.x < 0 || corner.y < 0 || corner.x >= image.cols || corner.y >= image.rows) {
                isFullyVisible = false;
                break;
            }
        }
        if (isFullyVisible) {
            fullyVisibleMarkers.push_back(corners);
        }
    }

    return fullyVisibleMarkers;
}


// Function to get the center of each tag
std::vector<Eigen::Vector2d> getTagCenters(const std::vector<std::vector<cv::Point2f>> &tagCorners) {
    std::vector<Eigen::Vector2d> centers;
    for (const auto &corners : tagCorners) {
        Eigen::Vector2d center(0, 0);
        for (const auto &corner : corners) {
            center += Eigen::Vector2d(corner.x, corner.y);
        }
        center /= corners.size();
        centers.push_back(center);
    }
    return centers;
}

// Function to load camera intrinsics and resolution from a YAML file
void loadCameraIntrinsicsAndResolution(const std::string &yamlFile, Eigen::Matrix3d &K, int &imageWidth, int &imageHeight) {
    YAML::Node config = YAML::LoadFile(yamlFile);
    K << config["fx"].as<double>(), 0, config["cx"].as<double>(),
         0, config["fy"].as<double>(), config["cy"].as<double>(),
         0, 0, 1;

    imageWidth = config["width"].as<int>();
    imageHeight = config["height"].as<int>();
}

// Function to project a 2D pixel into world space using depth = t_z - z_tag
Point pixelToWorld(const Eigen::Vector2d &pixel, const Eigen::Matrix3d &K, const Eigen::Affine3d &pose, double tagZ) {
    Eigen::Matrix3d R = pose.rotation();      // Camera rotation matrix
    Eigen::Vector3d t = pose.translation();  // Camera translation vector

    // Convert pixel to normalized camera coordinates
    Eigen::Vector3d normalizedPixel = K.inverse() * Eigen::Vector3d(pixel.x(), pixel.y(), 1.0);

    // Calculate depth as the difference between camera z and tag z
    double depth = abs(t.z() - tagZ);
    //double depth = 10;
    //std::cout << "depth = " << depth << std::endl;

    // Scale normalized coordinates by depth to get the 3D point in the camera frame
    Eigen::Vector3d pointInCameraFrame = normalizedPixel * depth;

    // Transform the point into the world frame
    Eigen::Vector3d pointInWorldFrame = R * pointInCameraFrame + t;

    return pointInWorldFrame;
}

// Function to project a 3D world point into 2D pixel space
Eigen::Vector2d worldToPixel(const Point &worldPoint, const Eigen::Matrix3d &K, const Eigen::Affine3d &pose) {
    Eigen::Matrix3d R = pose.rotation();      // Camera rotation matrix
    Eigen::Vector3d t = pose.translation();  // Camera translation vector

    // Transform the world point into the camera frame
    Eigen::Vector3d pointInCameraFrame = R.inverse() * (worldPoint - t);

    // Check if the point is in front of the camera
    if (pointInCameraFrame.z() <= 0) {
        throw std::runtime_error("Point is behind the camera and cannot be projected.");
    }

    // Project the point onto the normalized image plane
    Eigen::Vector3d normalizedPixel = pointInCameraFrame / pointInCameraFrame.z();

    // Convert normalized image coordinates to pixel space
    Eigen::Vector3d pixel = K * normalizedPixel;

    return Eigen::Vector2d(pixel.x(), pixel.y());
}

// writes the report file with backprojection(3D) and reprojection error(2D) components
void writeReport(const std::vector<double>& BPE_x,
                 const std::vector<double>& BPE_y,
                 const std::vector<double>& BPE_z,
                 const std::vector<double>& RPE_x,
                 const std::vector<double>& RPE_y,
                 const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening report file." << std::endl;
        return;
    }

    // Find maximum width for proper alignment
    int maxWidth = 0;
    std::vector<std::string> headers = {"BPE_x", "BPE_y", "BPE_z", "RPE_x", "RPE_y"};
    for (const auto& header : headers) {
        maxWidth = std::max(maxWidth, static_cast<int>(header.length()));
    }
    for (size_t i = 0; i < BPE_x.size(); ++i) {
        maxWidth = std::max(maxWidth, static_cast<int>(std::to_string(BPE_x[i]).length()));
        maxWidth = std::max(maxWidth, static_cast<int>(std::to_string(BPE_y[i]).length()));
        maxWidth = std::max(maxWidth, static_cast<int>(std::to_string(BPE_z[i]).length()));
        maxWidth = std::max(maxWidth, static_cast<int>(std::to_string(RPE_x[i]).length()));
        maxWidth = std::max(maxWidth, static_cast<int>(std::to_string(RPE_y[i]).length()));
    }

    // Set the width slightly larger than the max width for spacing
    maxWidth += 5;

    // Write the header with the correct spacing
    for (const auto& header : headers) {
        outFile << std::left << std::setw(maxWidth) << header;
    }
    outFile << '\n';

    // Write the vector data to the file with proper alignment
    for (size_t i = 0; i < BPE_x.size(); i++) {
        outFile << std::left << std::setw(maxWidth) << BPE_x[i]
                << std::setw(maxWidth) << BPE_y[i]
                << std::setw(maxWidth) << BPE_z[i]
                << std::setw(maxWidth) << RPE_x[i]
                << std::setw(maxWidth) << RPE_y[i] << '\n';
    }

    outFile.close();
    std::cout << "Error report successfully written to " << filename << std::endl;
}

void visualizePointsAndPoses(const std::vector<Point> &groundPoints,
                             const std::vector<Point> &backprojectedPoints,
                             const std::map<std::string, Eigen::Affine3d> &poses,
                             Eigen::Vector3d centerPoint) {
    pangolin::CreateWindowAndBind("Visualization: Points and Poses", 1024, 768);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(
            centerPoint.x(), centerPoint.y(), centerPoint.z() - 20, // Camera position
            centerPoint.x(), centerPoint.y(), centerPoint.z(),      // Look at the center point
            pangolin::AxisY)                                        // Up direction
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        // Draw ground truth points in green
        glColor3f(0.0f, 1.0f, 0.0f);
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (const auto &p : groundPoints) {
            glVertex3f(p.x(), p.y(), p.z());
        }
        glEnd();

        // Draw backprojected points in red
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (const auto &p : backprojectedPoints) {
            glVertex3f(p.x(), p.y(), p.z());
        }
        glEnd();

        // Draw camera poses as triads
        for (const auto &[imagePath, pose] : poses) {
            Eigen::Matrix3d R = pose.rotation();
            Eigen::Vector3d t = pose.translation();

            // Draw the triad at the pose
            glLineWidth(2.0f);
            glBegin(GL_LINES);

            // X-axis (red)
            glColor3f(1.0f, 0.0f, 0.0f);
            glVertex3f(t.x(), t.y(), t.z());
            Eigen::Vector3d xAxis = t + R.col(0);
            glVertex3f(xAxis.x(), xAxis.y(), xAxis.z());

            // Y-axis (green)
            glColor3f(0.0f, 1.0f, 0.0f);
            glVertex3f(t.x(), t.y(), t.z());
            Eigen::Vector3d yAxis = t + R.col(1);
            glVertex3f(yAxis.x(), yAxis.y(), yAxis.z());

            // Z-axis (blue)
            glColor3f(0.0f, 0.0f, 1.0f);
            glVertex3f(t.x(), t.y(), t.z());
            Eigen::Vector3d zAxis = t + R.col(2);
            glVertex3f(zAxis.x(), zAxis.y(), zAxis.z());

            glEnd();
        }

        pangolin::FinishFrame();
    }
}

// Function to display the image with tag centers highlighted
void displayImageWithTags(const cv::Mat &image, const std::vector<Eigen::Vector2d> &tagCenters) {
    cv::Mat displayImage = image.clone();

    // Draw the tag centers on the image
    for (const auto &center : tagCenters) {
        cv::circle(displayImage, cv::Point(center.x(), center.y()), 5, cv::Scalar(0, 0, 255), -1); // Red dot
    }

    // Show the image
    cv::imshow("Tag Detection", displayImage);
    cv::waitKey(30); // Display for 30ms
}

// returns true if image is blurry using laplacian variance
bool isImageBlurry(const cv::Mat& image) {
    // Check if the image is empty
    if (image.empty()) {
        std::cerr << "Invalid or unsupported image format!" << std::endl;
        return false;
    }

    cv::Mat gray, laplacian;
    // Check if the image is already grayscale
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        // Convert to grayscale
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        std::cerr << "Unsupported number of channels in image!" << std::endl;
        return false;
    }

    // Apply Laplacian function
    cv::Laplacian(gray, laplacian, CV_64F);

    // Calculate variance
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double variance = stddev.val[0] * stddev.val[0];

    // Threshold for determining blurriness
    double threshold = 500.0; // This threshold can be adjusted based on requirements
    //std::cout << variance << std::endl;

    return variance < threshold;
}

// Main function
int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <image_dir> <json_poses> <ground_points_file> <intrinsics_yaml> <output_file>" << std::endl;
        return 1;
    }

    std::string imageDir = argv[1];
    std::string jsonPosesFile = argv[2];
    std::string groundPointsFile = argv[3];
    std::string intrinsicsYaml = argv[4];
    std::string outputFile = argv[5];

    try {
        // Load image poses
        std::cout << "Loading poses from JSON file..." << std::endl;
        auto poses = loadPosesFromJson(jsonPosesFile);

        // Load ground points
        std::cout << "Loading ground points from file..." << std::endl;
        std::vector<Point> groundPoints;
        std::ifstream gpFile(groundPointsFile);
        if (!gpFile.is_open()) {
            throw std::runtime_error("Error: Unable to open ground points file: " + groundPointsFile);
        }
        double x, y, z;
        while (gpFile >> x >> y >> z) {
            groundPoints.emplace_back(x, y, z);
        }
        gpFile.close();

        // Pangolin visualization center with tag
        Eigen::Vector3d centerPoint(x, y, z); // Adjusted to initialize properly

        // Load camera intrinsics and resolution from YAML
        std::cout << "Loading camera intrinsics and resolution from YAML..." << std::endl;
        Eigen::Matrix3d K;
        int imageWidth, imageHeight;
        loadCameraIntrinsicsAndResolution(intrinsicsYaml, K, imageWidth, imageHeight);

        // Prepare ArUco dictionary
        cv::Ptr<cv::aruco::Dictionary> arucoDict = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11));

        // store backprojected points for visualization and L1 errors (meters)
        std::vector<Point> backprojectedPoints;
        std::vector<double> BPE_x;
        std::vector<double> BPE_y;
        std::vector<double> BPE_z;

        // store L1 reprojection errors (pixels)
        std::vector<double> RPE_x;
        std::vector<double> RPE_y;

        // process each image:pose pair
        int img_count = 0;
        for (const auto &[imagePath, pose] : poses) {
            std::string fullImagePath = imageDir + "/" + imagePath;
            cv::Mat image = cv::imread(fullImagePath, cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "Warning: Unable to load image: " << fullImagePath << std::endl;
                continue;
            }
            
            if (isImageBlurry(image)) {
                std::cerr << "Skipping blurry image: " << fullImagePath << std::endl;
                continue;
            }

            // Detect fully visible AprilTags
            std::vector<std::vector<cv::Point2f>> tagCorners = detectFullyVisibleAprilTags(image, arucoDict);
            if (tagCorners.empty()) {
                continue;
            }

            // Get the centers of the april tags
            std::vector<Eigen::Vector2d> tagCenters = getTagCenters(tagCorners);

            // Process each ground point
            for (const auto &gtPoint : groundPoints) {
                for (const auto &tagCenter : tagCenters) {
                    // Project the detected tag center to world space
                    Point projectedPoint = pixelToWorld(tagCenter, K, pose, gtPoint.z());
                    backprojectedPoints.push_back(projectedPoint);

                    // Compute and save the backprojection error components
                    double bpe_x = gtPoint.x() - projectedPoint.x();
                    double bpe_y = gtPoint.y() - projectedPoint.y();
                    double bpe_z = gtPoint.z() - projectedPoint.z();
                    BPE_x.push_back(bpe_x);
                    BPE_y.push_back(bpe_y);
                    BPE_z.push_back(bpe_z);

                    // Compute the reprojection error and save the components
                    Eigen::Vector2d reprojectedPixel = worldToPixel(gtPoint, K, pose);
                    double rpe_x = tagCenter.x() - reprojectedPixel.x();
                    double rpe_y = tagCenter.y() - reprojectedPixel.y();
                    RPE_x.push_back(rpe_x);
                    RPE_y.push_back(rpe_y);
                }
            }

            img_count++;
            std::cout << "Images processed: " << img_count << "/" << poses.size() << "\r" << std::flush;
        }

        // Save the backprojection and reprojection errors
        std::string report_name = outputFile + "_report.txt";
        writeReport(BPE_x, BPE_y, BPE_z, RPE_x, RPE_y, report_name);

        // Visualize the points
        visualizePointsAndPoses(groundPoints, backprojectedPoints, poses, centerPoint);

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}