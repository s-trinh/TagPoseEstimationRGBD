/*******************************************************************************
* Copyright 2019 s-trinh
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <iostream>

#include <visp3/core/vpConfig.h>

#if defined(VISP_HAVE_REALSENSE2)
#include <visp3/core/vpDisplay.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpPoint.h>
#include <visp3/core/vpPolygon.h>
#include <visp3/core/vpImageDraw.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/detection/vpDetectorAprilTag.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/mbt/vpMbtTukeyEstimator.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/vision/vpPose.h>

namespace {
static vpHomogeneousMatrix compute3d3dTransformation(const std::vector<vpPoint>& p, const std::vector<vpPoint>& q) {
    double N = p.size();

    vpColVector p_bar(3, 0.0);
    vpColVector q_bar(3, 0.0);
    for (size_t i = 0; i < p.size(); i++) {
        for (unsigned int j = 0; j < 3; j++) {
            p_bar[j] += p[i].oP[j];
            q_bar[j] += q[i].oP[j];
        }
    }

    for (unsigned int j = 0; j < 3; j++) {
        p_bar[j] /= N;
        q_bar[j] /= N;
    }

    vpMatrix pc(static_cast<unsigned int>(p.size()), 3);
    vpMatrix qc(static_cast<unsigned int>(q.size()), 3);

    for (unsigned int i = 0; i < static_cast<unsigned int>(p.size()); i++) {
        for (unsigned int j = 0; j < 3; j++) {
            pc[i][j] = p[i].oP[j] - p_bar[j];
            qc[i][j] = q[i].oP[j] - q_bar[j];
        }
    }

    vpMatrix pct_qc = pc.t()*qc;
    vpMatrix U = pct_qc, V;
    vpColVector W;
    U.svd(W, V);

    vpMatrix Vt = V.t();
    vpMatrix R = U*Vt;

    double det = R.det();
    if (det < 0) {
        Vt[2][0] *= -1;
        Vt[2][1] *= -1;
        Vt[2][2] *= -1;

        R = U*Vt;
    }

    vpColVector t = p_bar - R*q_bar;

    vpHomogeneousMatrix cMo;
    for (unsigned int i = 0; i < 3; i++) {
        for (unsigned int j = 0; j < 3; j++) {
            cMo[i][j] = R[i][j];
        }
        cMo[i][3] = t[i];
    }

    return cMo;
}

static void estimatePlaneEquationSVD(const std::vector<double> &point_cloud_face,
                                     vpColVector &plane_equation_estimated, vpColVector &centroid)
{
    const unsigned int max_iter = 10;
    double prev_error = 1e3;
    double error = 1e3 - 1;

    std::vector<double> weights(point_cloud_face.size() / 3, 1.0);
    std::vector<double> residues(point_cloud_face.size() / 3);
    vpMatrix M(static_cast<unsigned int>(point_cloud_face.size() / 3), 3);
    vpMbtTukeyEstimator<double> tukey;
    vpColVector normal;

    plane_equation_estimated.resize(4, false);
    for (unsigned int iter = 0; iter < max_iter && std::fabs(error - prev_error) > 1e-6; iter++) {
        if (iter != 0) {
            tukey.MEstimator(residues, weights, 1e-4);
        }

        // Compute centroid
        double centroid_x = 0.0, centroid_y = 0.0, centroid_z = 0.0;
        double total_w = 0.0;

        for (size_t i = 0; i < point_cloud_face.size() / 3; i++) {
            centroid_x += weights[i] * point_cloud_face[3 * i + 0];
            centroid_y += weights[i] * point_cloud_face[3 * i + 1];
            centroid_z += weights[i] * point_cloud_face[3 * i + 2];
            total_w += weights[i];
        }

        centroid_x /= total_w;
        centroid_y /= total_w;
        centroid_z /= total_w;

        // Minimization
        for (size_t i = 0; i < point_cloud_face.size() / 3; i++) {
            M[static_cast<unsigned int>(i)][0] = weights[i] * (point_cloud_face[3 * i + 0] - centroid_x);
            M[static_cast<unsigned int>(i)][1] = weights[i] * (point_cloud_face[3 * i + 1] - centroid_y);
            M[static_cast<unsigned int>(i)][2] = weights[i] * (point_cloud_face[3 * i + 2] - centroid_z);
        }

        vpColVector W;
        vpMatrix V;
        vpMatrix J = M.t() * M;
        J.svd(W, V);

        double smallestSv = W[0];
        unsigned int indexSmallestSv = 0;
        for (unsigned int i = 1; i < W.size(); i++) {
            if (W[i] < smallestSv) {
                smallestSv = W[i];
                indexSmallestSv = i;
            }
        }

        normal = V.getCol(indexSmallestSv);

        // Compute plane equation
        double A = normal[0], B = normal[1], C = normal[2];
        double D = -(A * centroid_x + B * centroid_y + C * centroid_z);

        // Update plane equation
        plane_equation_estimated[0] = A;
        plane_equation_estimated[1] = B;
        plane_equation_estimated[2] = C;
        plane_equation_estimated[3] = D;

        // Compute error points to estimated plane
        prev_error = error;
        error = 0.0;
        for (size_t i = 0; i < point_cloud_face.size() / 3; i++) {
            residues[i] = std::fabs(A * point_cloud_face[3 * i] + B * point_cloud_face[3 * i + 1] +
                    C * point_cloud_face[3 * i + 2] + D) /
                    sqrt(A * A + B * B + C * C);
            error += residues[i] * residues[i];
        }
        error /= sqrt(error / total_w);
    }

    // Update final weights
    tukey.MEstimator(residues, weights, 1e-4);

    // Update final centroid
    centroid.resize(3, false);
    double total_w = 0.0;

    for (size_t i = 0; i < point_cloud_face.size() / 3; i++) {
        centroid[0] += weights[i] * point_cloud_face[3 * i];
        centroid[1] += weights[i] * point_cloud_face[3 * i + 1];
        centroid[2] += weights[i] * point_cloud_face[3 * i + 2];
        total_w += weights[i];
    }

    centroid[0] /= total_w;
    centroid[1] /= total_w;
    centroid[2] /= total_w;

    // Compute final plane equation
    double A = normal[0], B = normal[1], C = normal[2];
    double D = -(A * centroid[0] + B * centroid[1] + C * centroid[2]);

    // Update final plane equation
    plane_equation_estimated[0] = A;
    plane_equation_estimated[1] = B;
    plane_equation_estimated[2] = C;
    plane_equation_estimated[3] = D;
}

static double computeZMethod1(const vpColVector& plane_equation, double x, double y) {
    return -plane_equation[3] / (plane_equation[0]*x + plane_equation[1]*y + plane_equation[2]);
}

static double computeZMethod2(const vpColVector& plane_equation, double x, double y) {
    double sum = plane_equation[0] * x + plane_equation[1] * y;
    return (-plane_equation[3] - sum) / plane_equation[2];
}

static bool validPose(const vpHomogeneousMatrix& cMo) {
    bool valid = true;

    for (unsigned int i = 0; i < cMo.getRows() && valid; i++) {
        for (unsigned int j = 0; j < cMo.getCols() && valid; j++) {
            if (vpMath::isNaN(cMo[i][j])) {
                valid = false;
            }
        }
    }

    return valid;
}
}

int main(int argc, char *argv[])
{
    bool save = false;
    double tagSize = 0.0415;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--save") {
            save = true;
        } else if (i+1 < argc && std::string(argv[i]) == "--tagSize") {
            tagSize = atof(argv[i+1]);
        }
    }

    std::cout << "save: " << save << std::endl;
    std::cout << "tagSize: " << tagSize << std::endl;

    std::string save_folder = "";
    if (save) {
        save_folder = vpTime::getDateTime("%Y-%m-%d_%H-%M-%S");
        std::cout << "save_folder: " << save_folder << std::endl;
        vpIoTools::makeDirectory(save_folder);
    }

    const int width = 640, height = 480;
    vpImage<vpRGBa> I_color(height, width);
    vpImage<unsigned char> I;
    vpImage<uint16_t> I_depth_raw(height, width);
    vpImage<vpRGBa> I_depth;

    // RealSense
    vpRealSense2 g;
    rs2::config config;
    config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGBA8, 30);
    config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, 30);
    g.open(config);

    const float depth_scale = g.getDepthScale();

    // Camera intrinsic parameters
    vpCameraParameters cam = g.getCameraParameters(rs2_stream::RS2_STREAM_COLOR);
    std::cout << "Cam:\n" << cam << std::endl;

    rs2::align align_to_color(RS2_STREAM_COLOR);
    g.acquire(reinterpret_cast<unsigned char *>(I_color.bitmap), reinterpret_cast<unsigned char *>(I_depth_raw.bitmap),
              NULL, NULL, &align_to_color);

    vpImage<vpRGBa> I_color2 = I_color;
    vpImage<vpRGBa> I_display(I_color.getHeight()*2, I_color.getWidth()*2);
    vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);
    vpDisplayX d(I_display, 0, 0, "RGBD pose fusion");

    // For pose estimation
    std::vector<vpPoint> pose_points;
    pose_points.push_back(vpPoint(-tagSize/2, -tagSize/2, 0));
    pose_points.push_back(vpPoint( tagSize/2, -tagSize/2, 0));
    pose_points.push_back(vpPoint( tagSize/2,  tagSize/2, 0));
    pose_points.push_back(vpPoint(-tagSize/2,  tagSize/2, 0));

    //AprilTag
    vpDetectorAprilTag::vpAprilTagFamily tagFamily = vpDetectorAprilTag::TAG_36h11;
    vpDetectorAprilTag::vpPoseEstimationMethod poseEstimationMethod = vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS;
    float quad_decimate = 1.0;
    int nThreads = 1;

    vpDetectorAprilTag detector(tagFamily);
    detector.setAprilTagQuadDecimate(quad_decimate);
    detector.setAprilTagPoseEstimationMethod(poseEstimationMethod);
    detector.setAprilTagNbThreads(nThreads);

    // Image results
    int cpt_frame = 0;

    while (true) {
        g.acquire(reinterpret_cast<unsigned char *>(I_color.bitmap), reinterpret_cast<unsigned char *>(I_depth_raw.bitmap),
                  NULL, NULL, &align_to_color);

        I_color2 = I_color;
        vpImageConvert::convert(I_color, I);
        vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);

        std::vector<vpHomogeneousMatrix> cMo_vec;
        detector.detect(I, tagSize, cam, cMo_vec);

        for (size_t i = 0; i < cMo_vec.size(); i++) {
            vpImageDraw::drawFrame(I_color, cMo_vec[i], cam, tagSize/2, vpColor::none, 3);
        }

        std::vector<std::vector<vpImagePoint> > tags_corners;
        tags_corners = detector.getPolygon();
        for (size_t i = 0; i < tags_corners.size(); i++) {
            vpPolygon polygon(tags_corners[i]);
            vpRect bb = polygon.getBoundingBox();
            unsigned int top = static_cast<unsigned int>(std::max( 0, static_cast<int>(bb.getTop()) ));
            unsigned int bottom = static_cast<unsigned int>(std::min( static_cast<int>(I.getHeight())-1, static_cast<int>(bb.getBottom()) ));
            unsigned int left = static_cast<unsigned int>(std::max( 0, static_cast<int>(bb.getLeft()) ));
            unsigned int right = static_cast<unsigned int>(std::min( static_cast<int>(I.getWidth())-1, static_cast<int>(bb.getRight()) ));

            std::vector<double> points_3d;
            points_3d.reserve( (bottom-top)*(right-left) );
            for (unsigned int idx_i = top; idx_i < bottom; idx_i++) {
                for (unsigned int idx_j = left; idx_j < right; idx_j++) {
                    vpImagePoint imPt(idx_i, idx_j);
                    if (I_depth_raw[idx_i][idx_j] > 0 && polygon.isInside(imPt)) {
                        double x = 0, y = 0;
                        vpPixelMeterConversion::convertPoint(cam, imPt.get_u(), imPt.get_v(), x, y);
                        double Z = I_depth_raw[idx_i][idx_j] * depth_scale;
                        points_3d.push_back(x*Z);
                        points_3d.push_back(y*Z);
                        points_3d.push_back(Z);
                    }
                }
            }

            if (points_3d.size() > 4) {
                std::vector<vpPoint> p, q;

                // Plane equation
                vpColVector plane_equation, centroid;
                estimatePlaneEquationSVD(points_3d, plane_equation, centroid);

                for (size_t j = 0; j < tags_corners[i].size(); j++) {
                    const vpImagePoint& imPt = tags_corners[i][j];
                    double x = 0, y = 0;
                    vpPixelMeterConversion::convertPoint(cam, imPt.get_u(), imPt.get_v(), x, y);
#define Z_METHOD1 1
#if Z_METHOD1
                    double Z = computeZMethod1(plane_equation, x, y);
                    if (Z < 0) {
                        Z = -Z;
                    }
                    p.push_back(vpPoint(x*Z, y*Z, Z));
#else
                    double depth = I_depth_raw[static_cast<int>(imPt.get_v())][static_cast<int>(imPt.get_u())]*depth_scale;
                    double X = x*depth;
                    double Y = y*depth;
                    double Z = computeZMethod2(plane_equation, X, Y);
                    p.push_back(vpPoint(X, Y, Z));
#endif

                    pose_points[j].set_x(x);
                    pose_points[j].set_y(y);
                }

                q.push_back(vpPoint(-tagSize/2, -tagSize/2, 0));
                q.push_back(vpPoint( tagSize/2, -tagSize/2, 0));
                q.push_back(vpPoint( tagSize/2,  tagSize/2, 0));
                q.push_back(vpPoint(-tagSize/2,  tagSize/2, 0));

                vpHomogeneousMatrix cMo = compute3d3dTransformation(p, q);

                if (validPose(cMo)) {
                    vpPose pose;
                    pose.addPoints(pose_points);
                    if (pose.computePose(vpPose::VIRTUAL_VS, cMo)) {
                        vpImageDraw::drawFrame(I_color2, cMo, cam, tagSize/2, vpColor::none, 3);
                    }
                }
            }
        }

        I_display.insert(I_depth, vpImagePoint(0, I_display.getWidth()/4));
        I_display.insert(I_color, vpImagePoint(I_depth.getHeight(), 0));
        I_display.insert(I_color2, vpImagePoint(I_depth.getHeight(), I_depth.getWidth()));

        vpDisplay::display(I_display);

        vpDisplay::displayText(I_display, 20+I_depth.getHeight(), 20, "Pose from homography + VVS", vpColor::red);
        vpDisplay::displayText(I_display, 20+I_depth.getHeight(), 20+I_depth.getWidth(), "Pose from RGBD fusion", vpColor::red);

        vpDisplay::flush(I_display);

        if (vpDisplay::getClick(I_display, false)) {
            break;
        }

        if (save) {
            char buffer[256];
            sprintf(buffer, "image_results_%04d.png", cpt_frame++);
            std::string filename = save_folder + "/" + buffer;
        }
    }

    return EXIT_SUCCESS;
}
#else
int main() {
    return 0;
}
#endif
