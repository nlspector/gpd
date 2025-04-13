#include <gpd/descriptor/image_generator.h>

// Need these includes for shadow calculation logic adapted from HandSet
// #include <boost/functional/hash.hpp> // Definition is included via hand_set.h
#include <boost/unordered_set.hpp>
#include <random> // For shadow point generation
#include <gpd/util/eigen_utils.h> // For slicing, etc.
#include <pcl/kdtree/kdtree_flann.h> // Added for KdTreeFLANN
#include <pcl/point_cloud.h> // Added for PointCloudRGBA
#include <pcl/common/impl/common.hpp> // Added for eigen conversion?
#include <pcl/point_types.h> // Added for PointXYZRGBA
#include <pcl/PCLPointCloud2.h> // Added for PCLPointCloud2
#include <pcl/conversions.h> // Added for conversions
#include <gpd/candidate/hand_set.h> // Include for boost::hash definition


// Define necessary types/helpers copied/adapted from HandSet.h/cpp
// boost::hash is defined in hand_set.h, included above

namespace gpd {
namespace descriptor {

// Vector3iEqual definition is needed if not included otherwise.
// Let's keep it here for now, but ideally it belongs in a shared utility header.
struct Vector3iEqual {
  inline bool operator()(const Eigen::Vector3i &a,
                         const Eigen::Vector3i &b) const {
    return a(0) == b(0) && a(1) == b(1) && a(2) == b(2);
  }
};

typedef boost::unordered_set<Eigen::Vector3i, boost::hash<Eigen::Vector3i>,
                             Vector3iEqual, std::allocator<Eigen::Vector3i>>
    Vector3iSet;

// Helper function adapted from HandSet::calculateShadowForCamera
namespace {
// Put helpers in anonymous namespace to limit scope
void calculateShadowForCameraLocal(const Eigen::Matrix3Xd &points,
                                const Eigen::Vector3d &shadow_vec,
                                int num_shadow_points, double voxel_grid_size,
                                Vector3iSet &shadow_set) {
    const int n = points.cols() * num_shadow_points;
    const double voxel_grid_size_mult = 1.0 / voxel_grid_size;
    const double max_rand = 1.0 / 32767.0; // From fastrand used in original code

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        const int pt_idx = i / num_shadow_points;
        shadow_set.insert(
            ((points.col(pt_idx) + (distrib(gen) * shadow_vec)) *
             voxel_grid_size_mult)
                .cast<int>());
    }
}

// Helper function adapted from HandSet::intersection
Vector3iSet intersectionLocal(const Vector3iSet &set1, const Vector3iSet &set2) {
    Vector3iSet result;
    if (set1.empty() || set2.empty()) return result;

    const Vector3iSet *smaller_set = &set1;
    const Vector3iSet *larger_set = &set2;
    if (set1.size() > set2.size()) {
        smaller_set = &set2;
        larger_set = &set1;
    }

    for (const auto& elem : *smaller_set) {
        if (larger_set->count(elem)) {
            result.insert(elem);
        }
    }
    return result;
}

// Helper function adapted from HandSet::shadowVoxelsToPoints
Eigen::Matrix3Xd shadowVoxelsToPointsLocal(
    const std::vector<Eigen::Vector3i> &voxels, double voxel_grid_size) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> distr{0.0, 1.0};
    Eigen::Matrix3Xd shadow(3, voxels.size());

    for (int i = 0; i < voxels.size(); i++) {
        shadow.col(i) =
            voxels[i].cast<double>() * voxel_grid_size +
            Eigen::Vector3d::Ones() * distr(gen) * voxel_grid_size * 0.3;
    }
    return shadow;
}


// Helper function adapted from HandSet::calculateShadow
// Takes original list for view points, nn_list for points to cast shadow from
Eigen::Matrix3Xd calculateShadowLocal(const util::PointList &original_list,
                                      const util::PointList &nn_list,
                                      double shadow_length) {
    const double voxel_grid_size = 0.003;
    double num_shadow_points = floor(shadow_length / voxel_grid_size);
    Eigen::Matrix3Xd final_shadow;

    // Use nn_list size for checking emptiness
    if (nn_list.size() == 0) {
        return final_shadow;
    }

    // Stricter check for camera info consistency *before* the loop
    const int num_cams = original_list.getCamSource().rows();
    const int num_view_points = original_list.getViewPoints().cols();

    if (num_cams <= 0 || num_view_points <= 0) {
        // Don't print warning if intentionally no cameras provided (num_cams == 0)
        if (num_cams != 0 || num_view_points != 0) {
             printf("Warning: No camera sources (%d) or view points (%d) available for shadow calculation.\n", num_cams, num_view_points);
        }
        return final_shadow; // Exit early if no cameras or views
    }

    if (num_cams != num_view_points) {
         printf("Warning: Mismatch between number of camera sources (%d) and view points (%d). Cannot calculate shadow reliably.\n", num_cams, num_view_points);
         // Decide whether to proceed partially or return. Returning seems safer.
         return final_shadow;
    }

    // If we reach here, num_cams == num_view_points > 0

    // Calculate center based on the neighborhood points (nn_list)
    Eigen::Vector3d center = nn_list.getPoints().rowwise().mean();
    std::vector<Vector3iSet> shadows_per_cam;
    shadows_per_cam.resize(num_cams, Vector3iSet(num_shadow_points * 1000));
    bool first_cam = true;
    Vector3iSet intersection_set;

    // Loop should now be safe regarding index 'i' for both matrices
    for (int i = 0; i < num_cams; i++) {
        if (original_list.getCamSource().row(i).any()) {
            // No need for the inner bounds check anymore due to the stricter check above
            Eigen::Vector3d shadow_vec = center - original_list.getViewPoints().col(i);
            if (shadow_vec.norm() > 1e-6) {
                 shadow_vec = shadow_length * shadow_vec.normalized();
            } else {
                 continue;
            }
            // Use nn_list's points for casting shadows
            calculateShadowForCameraLocal(nn_list.getPoints(), shadow_vec,
                                     num_shadow_points, voxel_grid_size, shadows_per_cam[i]);
            if (!shadows_per_cam[i].empty()) {
                if (first_cam) {
                    intersection_set = shadows_per_cam[i];
                    first_cam = false;
                } else {
                    intersection_set = intersectionLocal(intersection_set, shadows_per_cam[i]);
                }
            }
        }
    }

    std::vector<Eigen::Vector3i> final_voxels(intersection_set.begin(), intersection_set.end());
    final_shadow = shadowVoxelsToPointsLocal(final_voxels, voxel_grid_size);
    return final_shadow;
}
} // end anonymous namespace


ImageGenerator::ImageGenerator(const descriptor::ImageGeometry &image_geometry,
                               int num_threads, int num_orientations,
                               bool is_plotting, bool remove_plane)
    : image_params_(image_geometry),
      num_threads_(num_threads),
      num_orientations_(num_orientations),
      remove_plane_(remove_plane) {
  image_strategy_ = descriptor::ImageStrategy::makeImageStrategy(
      image_geometry, num_threads_, num_orientations_, is_plotting);
}

void ImageGenerator::createImages(
    const util::Cloud &cloud_cam,
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    std::vector<std::unique_ptr<cv::Mat>> &images_out,
    std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const {
  double t0 = omp_get_wtime();

  Eigen::Matrix3Xd points =
      cloud_cam.getCloudProcessed()->getMatrixXfMap().cast<double>().block(
          0, 0, 3, cloud_cam.getCloudProcessed()->points.size());
  util::PointList point_list(points, cloud_cam.getNormals(),
                             cloud_cam.getCameraSource(),
                             cloud_cam.getViewPoints());

  if (remove_plane_) {
    removePlane(cloud_cam, point_list);
  }

  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  // Use the *original* cloud for the kdtree, matching the constructor logic.
  // If remove_plane modified point_list, slicing below will use the modified indices.
  kdtree.setInputCloud(cloud_cam.getCloudProcessed());
  std::vector<int> nn_indices;
  std::vector<float> nn_dists;

  Eigen::Vector3d image_dims;
  image_dims << image_params_.depth_, image_params_.height_ / 2.0,
      image_params_.outer_diameter_;
  double radius = image_dims.maxCoeff();

  std::vector<util::PointList> nn_points_list;
  nn_points_list.resize(hand_set_list.size());

  double t_slice = omp_get_wtime();

#ifdef _OPENMP
#pragma omp parallel for private(nn_indices, nn_dists) num_threads(num_threads_)
#endif
  for (int i = 0; i < hand_set_list.size(); i++) {
    pcl::PointXYZRGBA sample_pcl;
    sample_pcl.getVector3fMap() = hand_set_list[i]->getSample().cast<float>();

    if (kdtree.radiusSearch(sample_pcl, radius, nn_indices, nn_dists) > 0) {
      // Slice the potentially plane-removed point_list
      nn_points_list[i] = point_list.slice(nn_indices);
    }
  }
  printf("neighborhoods search time: %3.4f\n", omp_get_wtime() - t_slice);

  createImageList(hand_set_list, nn_points_list, images_out, hands_out);
  printf("Created %zu images in %3.4fs\n", images_out.size(),
         omp_get_wtime() - t0);
}

void ImageGenerator::createImageList(
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const std::vector<util::PointList> &nn_points_list,
    std::vector<std::unique_ptr<cv::Mat>> &images_out,
    std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const {
  double t0_images = omp_get_wtime();

  int m = 0;
  if (!hand_set_list.empty()) {
      m = hand_set_list[0]->getHands().size();
  }
  // int n = hand_set_list.size() * m; // n not used
  std::vector<std::vector<std::unique_ptr<cv::Mat>>> images_list(hand_set_list.size());

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads_)
#endif
  for (int i = 0; i < hand_set_list.size(); i++) {
     images_list[i] =
         image_strategy_->createImages(*hand_set_list[i], nn_points_list[i]);
  }

  for (int i = 0; i < hand_set_list.size(); i++) {
    for (int j = 0; j < hand_set_list[i]->getHands().size(); j++) {
      if (hand_set_list[i]->getIsValid()(j)) {
             images_out.push_back(std::move(images_list[i][j]));
             hands_out.push_back(std::move(hand_set_list[i]->getHands()[j]));
        }
    }
  }
} // Correct closing brace for createImageList

// New function implementation - Placed correctly in namespace scope
std::unique_ptr<cv::Mat> ImageGenerator::createImage(
    const util::Cloud &cloud_cam,
    const candidate::Hand &hand) const {

  // 1. Prepare Original PointList (contains full view_points etc.)
  Eigen::Matrix3Xd points_orig =
      cloud_cam.getCloudProcessed()->getMatrixXfMap().cast<double>().block(
          0, 0, 3, cloud_cam.getCloudProcessed()->points.size());
  util::PointList original_point_list(points_orig, cloud_cam.getNormals(),
                                      cloud_cam.getCameraSource(),
                                      cloud_cam.getViewPoints());



  // 3. Prepare KD-Tree (using points from potentially modified working_point_list)
  pcl::KdTreeFLANN<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::KdTreeFLANN<pcl::PointXYZRGBA>());


  kdtree->setInputCloud(cloud_cam.getCloudProcessed());

  // 4. Find Neighborhood Points (nn_points) for the specific hand
  std::vector<int> nn_indices;
  std::vector<float> nn_dists;
  Eigen::Vector3d image_dims;
  image_dims << image_params_.depth_, image_params_.height_ / 2.0,
      image_params_.outer_diameter_;
  double radius = image_dims.maxCoeff();
  util::PointList nn_points; // This will be the sliced list from working_point_list

  pcl::PointXYZRGBA sample_pcl;
  sample_pcl.getVector3fMap() = hand.getSample().cast<float>();

  if (kdtree->radiusSearch(sample_pcl, radius, nn_indices, nn_dists) > 0) {
      if (!nn_indices.empty()) {
         // Slice the potentially plane-removed working_point_list
         nn_points = original_point_list.slice(nn_indices);
      }
  }

  if (nn_points.size() == 0) {
    return nullptr;
  }

  // 5. Calculate Shadow Points
  double shadow_length = image_params_.shadow_length_; // Use configured value
  Eigen::Matrix3Xd shadow_points = calculateShadowLocal(original_point_list, nn_points, shadow_length);

  // Ensure shadow_points is valid before passing, otherwise pass an empty matrix.
  const Eigen::Matrix3Xd& shadow_points_ref = (shadow_points.cols() > 0) ? shadow_points : Eigen::Matrix3Xd();


  // 6. Call the appropriate strategy's createImage method
  return image_strategy_->createImage(hand, nn_points, shadow_points_ref);
} // Correct closing brace for createImage


void ImageGenerator::removePlane(const util::Cloud &cloud_cam,
                                 util::PointList &point_list) const {
  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  if (!cloud_cam.getCloudProcessed() || cloud_cam.getCloudProcessed()->empty()) {
      printf("Warning: Cannot remove plane from empty or invalid cloud.\n");
      return;
  }
  seg.setInputCloud(cloud_cam.getCloudProcessed());
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.size() > 0) {
    pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
    extract.setInputCloud(cloud_cam.getCloudProcessed());
    extract.setIndices(inliers);
    extract.setNegative(true);
    std::vector<int> indices;
    extract.filter(indices); // Get indices of non-planar points relative to original cloud

    if (!indices.empty()) {
        // We need to map these original cloud indices to the current point_list indices,
        // which might be challenging if point_list was already modified (e.g., voxelized).
        // Assuming removePlane is called *before* significant modification or
        // that point_list indices correspond directly to cloud_processed indices.
        // A safer approach might involve re-filtering based on coordinates if indices mismatch.
        // For now, proceeding with the simpler index slicing, assuming consistency.
        try {
             // Check if indices are within bounds of the *original* cloud size implied by point_list source
             // This check is difficult without original size. Slicing might throw if indices are bad.
             util::PointList filtered_list = point_list.slice(indices);
             point_list = filtered_list; // Update point_list in place
             printf("Removed plane from point list. %d points remaining.\n", (int)point_list.size());
        } catch (const std::out_of_range& oor) {
            printf("Error slicing point list after plane removal (index out of range). Check index consistency. Skipping plane removal effect.\n");
        } catch (...) {
             printf("Unknown error slicing point list after plane removal. Skipping plane removal effect.\n");
        }

    } else {
      printf("Plane fit failed or resulted in empty cloud after extraction. Using original points ...\n");
    }
  } else {
     printf("No plane found for removal.\n");
  }
} // Correct closing brace for removePlane

}  // namespace descriptor
}  // namespace gpd
// Ensure final closing braces for namespaces are present
