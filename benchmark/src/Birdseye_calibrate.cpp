#include <opencv2/opencv.hpp>
#include <apriltag.h>
#include <tag36h11.h>
#include <filesystem>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

bool loadCameraIntrinsics(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    YAML::Node config = YAML::LoadFile(filename);
    try {
        double fx = config["fx"].as<double>();
        double fy = config["fy"].as<double>();
        double cx = config["cx"].as<double>();
        double cy = config["cy"].as<double>();
        double k1 = config["k1"].as<double>();
        double k2 = config["k2"].as<double>();
        double p1 = config["p1"].as<double>();
        double p2 = config["p2"].as<double>();

        cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0, cx,
                                                 0, fy, cy,
                                                 0, 0, 1);
        distCoeffs = (cv::Mat_<double>(4,1) << k1, k2, p1, p2);
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load or parse camera intrinsics from YAML file: " << e.what() << std::endl;
        return false;
    }
}

// Function to process a single image for AprilTag detection
void processImage(cv::Mat& image, apriltag_detector_t* td, const std::vector<cv::Point3f>& objp,
                  std::vector<std::vector<cv::Point3f>>& objectPoints, std::vector<std::vector<cv::Point2f>>& imagePoints) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detect AprilTags
    image_u8_t im = {.width = gray.cols, .height = gray.rows, .stride = gray.cols, .buf = gray.data};
    zarray_t* detections = apriltag_detector_detect(td, &im);

    // Gather points for calibration if tags are detected
    if (zarray_size(detections) > 0) {
        std::cout << "tag detected" << std::endl;
        std::vector<cv::Point2f> imageCorners;
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t* det;
            zarray_get(detections, i, &det);
            imageCorners.push_back(cv::Point2f(det->p[3][0], det->p[3][1])); // bottom-left
            imageCorners.push_back(cv::Point2f(det->p[2][0], det->p[2][1])); // bottom-right
            imageCorners.push_back(cv::Point2f(det->p[1][0], det->p[1][1])); // top-right
            imageCorners.push_back(cv::Point2f(det->p[0][0], det->p[0][1])); // top-left
        }
        objectPoints.push_back(objp);
        imagePoints.push_back(imageCorners);
    }

    apriltag_detections_destroy(detections);
}

// Function to select a random subset and return a copy
std::pair<std::vector<std::vector<cv::Point3f>>, std::vector<std::vector<cv::Point2f>>>
selectRandomSubset(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                   const std::vector<std::vector<cv::Point2f>>& imagePoints,
                   int num_samples) {
    std::random_device rd;
    std::mt19937 eng(rd());
    std::vector<int> indices(objectPoints.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), eng);

    std::vector<std::vector<cv::Point3f>> selectedObjPoints;
    std::vector<std::vector<cv::Point2f>> selectedImgPoints;

    for (int i = 0; i < num_samples; ++i) {
        selectedObjPoints.push_back(objectPoints[indices[i]]);
        selectedImgPoints.push_back(imagePoints[indices[i]]);
    }

    return {selectedObjPoints, selectedImgPoints};
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
    double threshold = 1000.0; // This threshold can be adjusted based on requirements

    return variance < threshold;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <yaml_intrinsics_file>" << std::endl;
        return -1;
    }

    std::string imageDir = argv[1];
    std::string yamlFile = argv[2];

    // Initialize AprilTag structures
    apriltag_family_t* tf = tag36h11_create();
    apriltag_detector_t* td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);

    // Load initial camera intrinsics
    cv::Mat cameraMatrix, distCoeffs;
    if (!loadCameraIntrinsics(yamlFile, cameraMatrix, distCoeffs)) {
        return -1;
    }

    // Define object points based on the real-world dimensions of the AprilTags
    float tagSize = 0.50; // Tag size in meters
    std::vector<cv::Point3f> objp = {
        {0, 0, 0},
        {tagSize, 0, 0},
        {tagSize, tagSize, 0},
        {0, tagSize, 0}
    };

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    // Iterate over each image file in the directory
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        cv::Mat image = cv::imread(entry.path().string());
        if (!image.empty() && !isImageBlurry(image)) {
            processImage(image, td, objp, objectPoints, imagePoints);
        }
    }

    // Perform camera calibration if enough image points have been collected
    auto [subsetObjectPoints, subsetImagePoints] = selectRandomSubset(objectPoints, imagePoints, 100);

    if (!subsetImagePoints.empty()) {
        //cv::Mat cameraMatrix, distCoeffs;
        std::vector<cv::Mat> rvecs, tvecs;
        cv::calibrateCamera(subsetObjectPoints, subsetImagePoints, cv::Size(1920, 1200), cameraMatrix, distCoeffs, rvecs, tvecs, cv::CALIB_USE_INTRINSIC_GUESS);  // Use the actual size of your images

        // Output camera matrix
        std::cout << "Camera Matrix:" << std::endl << cameraMatrix << std::endl;
    }

    // Cleanup
    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);
    cv::destroyAllWindows();

    return 0;
}
