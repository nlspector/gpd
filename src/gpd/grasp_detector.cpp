#include <gpd/grasp_detector.h>
#include <gpd/util/plot.h>

#include <omp.h>
#include <Eigen/Dense>

namespace gpd {

GraspDetector::GraspDetector(const std::string &config_filename) {
  Eigen::initParallel();

  // Read parameters from configuration file.
  util::ConfigFile config_file(config_filename);
  config_file.ExtractKeys();

  // Read hand geometry parameters.
  std::string hand_geometry_filename =
      config_file.getValueOfKeyAsString("hand_geometry_filename", "");
  if (hand_geometry_filename == "0") {
    hand_geometry_filename = config_filename;
  }
  candidate::HandGeometry hand_geom(hand_geometry_filename);
  std::cout << hand_geom;

  // Read plotting parameters.
  plot_normals_ = config_file.getValueOfKey<bool>("plot_normals", false);
  plot_samples_ = config_file.getValueOfKey<bool>("plot_samples", true);
  plot_candidates_ = config_file.getValueOfKey<bool>("plot_candidates", false);
  plot_filtered_candidates_ =
      config_file.getValueOfKey<bool>("plot_filtered_candidates", false);
  plot_valid_grasps_ =
      config_file.getValueOfKey<bool>("plot_valid_grasps", false);
  plot_clustered_grasps_ =
      config_file.getValueOfKey<bool>("plot_clustered_grasps", false);
  plot_selected_grasps_ =
      config_file.getValueOfKey<bool>("plot_selected_grasps", false);
  printf("============ PLOTTING ========================\n");
  printf("plot_normals: %s\n", plot_normals_ ? "true" : "false");
  printf("plot_samples %s\n", plot_samples_ ? "true" : "false");
  printf("plot_candidates: %s\n", plot_candidates_ ? "true" : "false");
  printf("plot_filtered_candidates: %s\n",
         plot_filtered_candidates_ ? "true" : "false");
  printf("plot_valid_grasps: %s\n", plot_valid_grasps_ ? "true" : "false");
  printf("plot_clustered_grasps: %s\n",
         plot_clustered_grasps_ ? "true" : "false");
  printf("plot_selected_grasps: %s\n",
         plot_selected_grasps_ ? "true" : "false");
  printf("==============================================\n");

  // Create object to generate grasp candidates.
  candidate::CandidatesGenerator::Parameters generator_params;
  generator_params.num_samples_ =
      config_file.getValueOfKey<int>("num_samples", 1000);
  generator_params.num_threads_ =
      config_file.getValueOfKey<int>("num_threads", 1);
  generator_params.remove_statistical_outliers_ =
      config_file.getValueOfKey<bool>("remove_outliers", false);
  generator_params.sample_above_plane_ =
      config_file.getValueOfKey<bool>("sample_above_plane", false);
  generator_params.voxelize_ =
      config_file.getValueOfKey<bool>("voxelize", true);
  generator_params.voxel_size_ =
      config_file.getValueOfKey<double>("voxel_size", 0.003);
  generator_params.normals_radius_ =
      config_file.getValueOfKey<double>("normals_radius", 0.03);
  generator_params.refine_normals_k_ =
      config_file.getValueOfKey<int>("refine_normals_k", 0);
  generator_params.workspace_ =
      config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");

  candidate::HandSearch::Parameters hand_search_params;
  hand_search_params.hand_geometry_ = hand_geom;
  hand_search_params.nn_radius_frames_ =
      config_file.getValueOfKey<double>("nn_radius", 0.01);
  hand_search_params.num_samples_ =
      config_file.getValueOfKey<int>("num_samples", 1000);
  hand_search_params.num_threads_ =
      config_file.getValueOfKey<int>("num_threads", 1);
  hand_search_params.num_orientations_ =
      config_file.getValueOfKey<int>("num_orientations", 8);
  hand_search_params.num_finger_placements_ =
      config_file.getValueOfKey<int>("num_finger_placements", 10);
  hand_search_params.deepen_hand_ =
      config_file.getValueOfKey<bool>("deepen_hand", true);
  hand_search_params.hand_axes_ =
      config_file.getValueOfKeyAsStdVectorInt("hand_axes", "2");
  hand_search_params.friction_coeff_ =
      config_file.getValueOfKey<double>("friction_coeff", 20.0);
  hand_search_params.min_viable_ =
      config_file.getValueOfKey<int>("min_viable", 6);
  candidates_generator_ = std::make_unique<candidate::CandidatesGenerator>(
      generator_params, hand_search_params);

  printf("============ CLOUD PREPROCESSING =============\n");
  printf("voxelize: %s\n", generator_params.voxelize_ ? "true" : "false");
  printf("voxel_size: %.3f\n", generator_params.voxel_size_);
  printf("remove_outliers: %s\n",
         generator_params.remove_statistical_outliers_ ? "true" : "false");
  printStdVector(generator_params.workspace_, "workspace");
  printf("sample_above_plane: %s\n",
         generator_params.sample_above_plane_ ? "true" : "false");
  printf("normals_radius: %.3f\n", generator_params.normals_radius_);
  printf("refine_normals_k: %d\n", generator_params.refine_normals_k_);
  printf("==============================================\n");

  printf("============ CANDIDATE GENERATION ============\n");
  printf("num_samples: %d\n", hand_search_params.num_samples_);
  printf("num_threads: %d\n", hand_search_params.num_threads_);
  printf("nn_radius: %3.2f\n", hand_search_params.nn_radius_frames_);
  printStdVector(hand_search_params.hand_axes_, "hand axes");
  printf("num_orientations: %d\n", hand_search_params.num_orientations_);
  printf("num_finger_placements: %d\n",
         hand_search_params.num_finger_placements_);
  printf("deepen_hand: %s\n",
         hand_search_params.deepen_hand_ ? "true" : "false");
  printf("friction_coeff: %3.2f\n", hand_search_params.friction_coeff_);
  printf("min_viable: %d\n", hand_search_params.min_viable_);
  printf("==============================================\n");

  // TODO: Set the camera position.
  //  Eigen::Matrix3Xd view_points(3,1);
  //  view_points << camera_position[0], camera_position[1], camera_position[2];

  // Read grasp image parameters.
  std::string image_geometry_filename =
      config_file.getValueOfKeyAsString("image_geometry_filename", "");
  if (image_geometry_filename == "0") {
    image_geometry_filename = config_filename;
  }
  descriptor::ImageGeometry image_geom(image_geometry_filename);
  std::cout << image_geom;

  // Read classification parameters and create classifier.
  std::string model_file = config_file.getValueOfKeyAsString("model_file", "");
  std::string weights_file =
      config_file.getValueOfKeyAsString("weights_file", "");
  if (!model_file.empty() || !weights_file.empty()) {
    int device = config_file.getValueOfKey<int>("device", 0);
    int batch_size = config_file.getValueOfKey<int>("batch_size", 1);
    classifier_ = net::Classifier::create(
        model_file, weights_file, static_cast<net::Classifier::Device>(device),
        batch_size);
    min_score_ = config_file.getValueOfKey<int>("min_score", 0);
    printf("============ CLASSIFIER ======================\n");
    printf("model_file: %s\n", model_file.c_str());
    printf("weights_file: %s\n", weights_file.c_str());
    printf("batch_size: %d\n", batch_size);
    printf("==============================================\n");
  }

  // Read additional grasp image creation parameters.
  bool remove_plane = config_file.getValueOfKey<bool>(
      "remove_plane_before_image_calculation", false);

  // Create object to create grasp images from grasp candidates (used for
  // classification).
  image_generator_ = std::make_unique<descriptor::ImageGenerator>(
      image_geom, hand_search_params.num_threads_,
      hand_search_params.num_orientations_, false, remove_plane);

  // Read grasp filtering parameters based on robot workspace and gripper width.
  workspace_grasps_ = config_file.getValueOfKeyAsStdVectorDouble(
      "workspace_grasps", "-1 1 -1 1 -1 1");
  min_aperture_ = config_file.getValueOfKey<double>("min_aperture", 0.0);
  max_aperture_ = config_file.getValueOfKey<double>("max_aperture", 0.085);
  printf("============ CANDIDATE FILTERING =============\n");
  printStdVector(workspace_grasps_, "candidate_workspace");
  printf("min_aperture: %3.4f\n", min_aperture_);
  printf("max_aperture: %3.4f\n", max_aperture_);
  printf("==============================================\n");

  // Read grasp filtering parameters based on approach direction.
  filter_approach_direction_ =
      config_file.getValueOfKey<bool>("filter_approach_direction", false);
  std::vector<double> approach =
      config_file.getValueOfKeyAsStdVectorDouble("direction", "1 0 0");
  direction_ << approach[0], approach[1], approach[2];
  thresh_rad_ = config_file.getValueOfKey<double>("thresh_rad", 2.3);

  // Read clustering parameters.
  int min_inliers = config_file.getValueOfKey<int>("min_inliers", 1);
  clustering_ = std::make_unique<Clustering>(min_inliers);
  cluster_grasps_ = min_inliers > 0 ? true : false;
  printf("============ CLUSTERING ======================\n");
  printf("min_inliers: %d\n", min_inliers);
  printf("==============================================\n\n");

  // Read grasp selection parameters.
  num_selected_ = config_file.getValueOfKey<int>("num_selected", 100);

  // Read second gripper parameters
  second_gripper_offset_.x() = config_file.getValueOfKey<double>("second_gripper_translation_x", 0.0);
  second_gripper_offset_.y() = config_file.getValueOfKey<double>("second_gripper_translation_y", 0.0);
  second_gripper_offset_.z() = config_file.getValueOfKey<double>("second_gripper_translation_z", 0.0);

  // Create plotter.
  plotter_ = std::make_unique<util::Plot>(hand_search_params.hand_axes_.size(),
                                          hand_search_params.num_orientations_);
}

std::vector<std::unique_ptr<candidate::Hand>> GraspDetector::detectGrasps(
    const util::Cloud &cloud, const util::Cloud &stem_cloud) {
  double t0_total = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  const candidate::HandGeometry &hand_geom =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  // Check if the point cloud is empty.
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("ERROR: Point cloud is empty!");
    hands_out.resize(0);
    return hands_out;
  }

  // Plot samples/indices.
  if (plot_samples_) {
    if (cloud.getSamples().cols() > 0) {
      plotter_->plotSamples(cloud.getSamples(), cloud.getCloudProcessed());
    } else if (cloud.getSampleIndices().size() > 0) {
      plotter_->plotSamples(cloud.getSampleIndices(),
                            cloud.getCloudProcessed());
    }
  }

  if (plot_normals_) {
    std::cout << "Plotting normals for different camera sources\n";
    plotter_->plotNormals(cloud);
  }

  // 1. Generate grasp candidates on the primary cloud.
  double t0_candidates = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      candidates_generator_->generateGraspCandidateSets(cloud);
  printf("Generated %zu primary hand sets.\n", hand_set_list.size());
  if (hand_set_list.size() == 0) {
    return hands_out;
  }
  double t_candidates = omp_get_wtime() - t0_candidates;
  if (plot_candidates_) {
    plotter_->plotFingers3D(hand_set_list, cloud.getCloudOriginal(),
                            "Grasp candidates", hand_geom);
  }

  // 2. Filter the primary candidates.
  double t0_filter = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_filtered =
      filterGraspsWorkspace(hand_set_list, workspace_grasps_);
  if (hand_set_list_filtered.size() == 0) {
    return hands_out;
  }
  if (plot_filtered_candidates_) {
    plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                            "Filtered Grasps (Aperture, Workspace)", hand_geom);
  }
  if (filter_approach_direction_) {
    hand_set_list_filtered =
        filterGraspsDirection(hand_set_list_filtered, direction_, thresh_rad_);
    if (plot_filtered_candidates_) {
      plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                              "Filtered Grasps (Approach)", hand_geom);
    }
  }
  double t_filter = omp_get_wtime() - t0_filter;
  if (hand_set_list_filtered.size() == 0) {
    return hands_out;
  }

  // 3. Create grasp descriptors (images) for primary hands.
  double t0_images = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> primary_images;
  image_generator_->createImages(cloud, hand_set_list_filtered, primary_images, hands);
  double t_images = omp_get_wtime() - t0_images;

  if (hands.empty()) {
      printf("No valid primary hands found after image generation.\n");
      return hands_out;
  }

  // 4. Classify the primary grasp candidates.
  double t0_classify = omp_get_wtime();
  std::vector<float> primary_scores = classifier_->classifyImages(primary_images);
  double t_classify = omp_get_wtime() - t0_classify;

  // 5. Evaluate stem grasps and calculate average scores.
  double t0_stem_eval = omp_get_wtime();

  // ---> Preprocess the stem cloud <--- 
  util::Cloud processed_stem_cloud = stem_cloud; // Create a mutable copy
  candidates_generator_->preprocessPointCloud(processed_stem_cloud);
  printf("Preprocessed stem cloud (%zu points remaining).\n", processed_stem_cloud.getCloudProcessed()->size());
  // Check if stem cloud is valid after processing
  if (processed_stem_cloud.getCloudProcessed()->empty()) {
      printf("Warning: Stem cloud is empty after preprocessing. Cannot evaluate stem grasps.\n");
      // Handle this case: perhaps return only primary grasps or empty list?
      // For now, continue but expect no stem images/scores.
      hands.clear(); // Clear hands as we can't calculate averaged scores
  }
  // <--- End stem cloud preprocessing --->

  std::vector<float> final_scores;
  final_scores.reserve(hands.size());

  // Prepare for batch processing of stem images if single image classification isn't efficient/available
  std::vector<std::unique_ptr<cv::Mat>> stem_images;
  std::vector<int> valid_hand_indices; // Keep track of hands for which stem images could be generated
  stem_images.reserve(hands.size());
  valid_hand_indices.reserve(hands.size());

  for (int i = 0; i < hands.size(); ++i) {
      const auto& primary_hand = hands[i];

      // Calculate stem hand pose
      Eigen::Matrix3d stem_orientation = primary_hand->getOrientation(); // Keep same orientation

      // Create a Hand object for the stem pose using primary hand's geometry/finger info.
      // The actual position will be implicitly handled by the image generation using the stem cloud
      // and the primary hand's sample point, combined with the offset applied during evaluation.
      // We use a valid constructor that takes sample, frame, and FingerHand.
      candidate::Hand stem_hand(primary_hand->getSample(), // Use primary sample initially
                                stem_orientation,        // Stem orientation
                                primary_hand->getFingerHand()); // Re-use primary FingerHand geometry

      // ---> Set the correct sample point incorporating the offset <--- 
      stem_hand.setSample(primary_hand->getSample() + primary_hand->getOrientation() * second_gripper_offset_);

      // Set the score to 0 initially (or keep existing constructor's default)
      stem_hand.setScore(0.0f);
      // NOTE: We don't explicitly set the stem_hand position, as it's not directly used by
      // createImage, which relies on the sample + orientation + cloud data.
      // The offset is handled conceptually by using stem_cloud with primary_hand's sample reference.

      // Generate image for the stem hand using the *processed* stem_cloud.
      std::unique_ptr<cv::Mat> stem_image = image_generator_->createImage(cloud, stem_hand);

      if (stem_image && !stem_image->empty()) {
          stem_images.push_back(std::move(stem_image));
          valid_hand_indices.push_back(i); // Store index of the original hand
      }
      // If stem image creation fails, we might skip this hand or assign a default low score.
      // For now, we just won't have a stem score for it.
  }

  std::vector<float> stem_scores;
  if (!stem_images.empty()) {
     stem_scores = classifier_->classifyImages(stem_images);
  }

  std::vector<std::unique_ptr<candidate::Hand>> valid_hands; // New list for hands with valid avg scores
  valid_hands.reserve(valid_hand_indices.size());

  for (int j = 0; j < valid_hand_indices.size(); ++j) {
      int original_index = valid_hand_indices[j];
      float primary_score = primary_scores[original_index];
      float stem_score = stem_scores[j]; // Scores correspond to the order of stem_images

      float final_score = (primary_score + stem_score) / 2.0f;
      printf("primary_score: %f, stem_score: %f, final_score: %f\n", primary_score, stem_score, final_score);

      hands[original_index]->setScore(final_score);
      valid_hands.push_back(std::move(hands[original_index])); // Move hand to the valid list
  }

  // Replace original hands list with the list of hands that have valid averaged scores
  hands = std::move(valid_hands);
  double t_stem_eval = omp_get_wtime() - t0_stem_eval;

  if (hands.empty()) {
      printf("No hands remained after stem evaluation and averaging.\n");
      return hands_out;
  }

  printf("Evaluated stem grasps and calculated average scores for %zu hands.\n", hands.size());

  // 6. Select the <num_selected> highest scoring grasps (using averaged scores).
  hands = selectGrasps(hands);
  if (plot_valid_grasps_) {
    // Plotting might need adjustment if we want to show both grippers
    plotter_->plotFingers3D(hands, cloud.getCloudOriginal(), "Valid Grasps (Avg Score)",
                            hand_geom);
  }

  // 7. Cluster the grasps (using averaged scores).
  double t0_cluster = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> clusters;
  if (cluster_grasps_) {
    clusters = clustering_->findClusters(hands);
    printf("Found %d clusters.\n", (int)clusters.size());
    if (clusters.size() <= 3) {
      printf(
          "Not enough clusters found! Adding all grasps from previous step.");
      for (int i = 0; i < hands.size(); i++) {
        clusters.push_back(std::move(hands[i]));
      }
    }
    if (plot_clustered_grasps_) {
      plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(),
                              "Clustered Grasps", hand_geom);
    }
  } else {
    clusters = std::move(hands);
  }
  double t_cluster = omp_get_wtime() - t0_cluster;

  // 8. Sort grasps by their averaged score.
  std::sort(clusters.begin(), clusters.end(), isScoreGreater);
  printf("======== Selected grasps ========\n");
  for (int i = 0; i < clusters.size(); i++) {
    std::cout << "Grasp " << i << ": " << clusters[i]->getScore() << "\n";
  }
  printf("Selected the %d best grasps.\n", (int)clusters.size());
  double t_total = omp_get_wtime() - t0_total;

  printf("======== RUNTIMES ========\n");
  printf(" 1. Candidate generation: %3.4fs\n", t_candidates);
  printf(" 2. Primary descriptor extraction: %3.4fs\n", t_images);
  printf(" 3. Primary classification: %3.4fs\n", t_classify);
  printf(" 4. Stem evaluation (img+classify): %3.4fs\n", t_stem_eval);
  // printf(" Filtering: %3.4fs\n", t_filter);
  // printf(" Clustering: %3.4fs\n", t_cluster);
  printf("==========\n");
  printf(" TOTAL: %3.4fs\n", t_total);

  if (plot_selected_grasps_) {
    plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(),
                            "Selected Grasps", hand_geom, false);
  }

  return clusters;
}

void GraspDetector::preprocessPointCloud(util::Cloud &cloud) {
  candidates_generator_->preprocessPointCloud(cloud);
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::filterGraspsWorkspace(
    std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const std::vector<double> &workspace) const {
  int remaining = 0;
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
  printf("Filtering grasps outside of workspace ...\n");

  const candidate::HandGeometry &hand_geometry =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<std::unique_ptr<candidate::Hand>> &hands =
        hand_set_list[i]->getHands();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i]->getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (!is_valid(j)) {
        continue;
      }
      double half_width = 0.5 * hand_geometry.outer_diameter_;
      Eigen::Vector3d left_bottom =
          hands[j]->getPosition() + half_width * hands[j]->getBinormal();
      Eigen::Vector3d right_bottom =
          hands[j]->getPosition() - half_width * hands[j]->getBinormal();
      Eigen::Vector3d left_top =
          left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
      Eigen::Vector3d right_top =
          left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
      Eigen::Vector3d approach =
          hands[j]->getPosition() - 0.05 * hands[j]->getApproach();
      Eigen::VectorXd x(5), y(5), z(5);
      x << left_bottom(0), right_bottom(0), left_top(0), right_top(0),
          approach(0);
      y << left_bottom(1), right_bottom(1), left_top(1), right_top(1),
          approach(1);
      z << left_bottom(2), right_bottom(2), left_top(2), right_top(2),
          approach(2);

      // Ensure the object fits into the hand and avoid grasps outside the
      // workspace.
      if (hands[j]->getGraspWidth() >= min_aperture_ &&
          hands[j]->getGraspWidth() <= max_aperture_ &&
          x.minCoeff() >= workspace[0] && x.maxCoeff() <= workspace[1] &&
          y.minCoeff() >= workspace[2] && y.maxCoeff() <= workspace[3] &&
          z.minCoeff() >= workspace[4] && z.maxCoeff() <= workspace[5]) {
        is_valid(j) = true;
        remaining++;
      } else {
        is_valid(j) = false;
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(std::move(hand_set_list[i]));
      hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
    }
  }

  printf("Number of grasp candidates within workspace and gripper width: %d\n",
         remaining);

  return hand_set_list_out;
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::generateGraspCandidates(const util::Cloud &cloud) {
  return candidates_generator_->generateGraspCandidateSets(cloud);
}

std::vector<std::unique_ptr<candidate::Hand>> GraspDetector::selectGrasps(
    std::vector<std::unique_ptr<candidate::Hand>> &hands) const {
  printf("Selecting the %d highest scoring grasps ...\n", num_selected_);

  int middle = std::min((int)hands.size(), num_selected_);
  std::partial_sort(hands.begin(), hands.begin() + middle, hands.end(),
                    isScoreGreater);
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  for (int i = 0; i < middle; i++) {
    hands_out.push_back(std::move(hands[i]));
    printf(" grasp #%d, score: %3.4f\n", i, hands_out[i]->getScore());
  }

  return hands_out;
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::filterGraspsDirection(
    std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const Eigen::Vector3d &direction, const double thresh_rad) {
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
  int remaining = 0;

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<std::unique_ptr<candidate::Hand>> &hands =
        hand_set_list[i]->getHands();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i]->getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (is_valid(j)) {
        double angle = acos(direction.transpose() * hands[j]->getApproach());
        if (angle > thresh_rad) {
          is_valid(j) = false;
        } else {
          remaining++;
        }
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(std::move(hand_set_list[i]));
      hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
    }
  }

  printf("Number of grasp candidates with correct approach direction: %d\n",
         remaining);

  return hand_set_list_out;
}

bool GraspDetector::createGraspImages(
    util::Cloud &cloud,
    std::vector<std::unique_ptr<candidate::Hand>> &hands_out,
    std::vector<std::unique_ptr<cv::Mat>> &images_out) {
  // Check if the point cloud is empty.
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("ERROR: Point cloud is empty!");
    hands_out.resize(0);
    images_out.resize(0);
    return false;
  }

  // Plot samples/indices.
  if (plot_samples_) {
    if (cloud.getSamples().cols() > 0) {
      plotter_->plotSamples(cloud.getSamples(), cloud.getCloudProcessed());
    } else if (cloud.getSampleIndices().size() > 0) {
      plotter_->plotSamples(cloud.getSampleIndices(),
                            cloud.getCloudProcessed());
    }
  }

  if (plot_normals_) {
    std::cout << "Plotting normals for different camera sources\n";
    plotter_->plotNormals(cloud);
  }

  // 1. Generate grasp candidates.
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      candidates_generator_->generateGraspCandidateSets(cloud);
  printf("Generated %zu hand sets.\n", hand_set_list.size());
  if (hand_set_list.size() == 0) {
    hands_out.resize(0);
    images_out.resize(0);
    return false;
  }

  const candidate::HandGeometry &hand_geom =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  // 2. Filter the candidates.
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_filtered =
      filterGraspsWorkspace(hand_set_list, workspace_grasps_);
  if (plot_filtered_candidates_) {
    plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                            "Filtered Grasps (Aperture, Workspace)", hand_geom);
  }
  if (filter_approach_direction_) {
    hand_set_list_filtered =
        filterGraspsDirection(hand_set_list_filtered, direction_, thresh_rad_);
    if (plot_filtered_candidates_) {
      plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                              "Filtered Grasps (Approach)", hand_geom);
    }
  }

  // 3. Create grasp descriptors (images).
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list_filtered, images_out,
                                 hands_out);

  return true;
}

std::vector<int> GraspDetector::evalGroundTruth(
    const util::Cloud &cloud_gt,
    std::vector<std::unique_ptr<candidate::Hand>> &hands) {
  return candidates_generator_->reevaluateHypotheses(cloud_gt, hands);
}

std::vector<std::unique_ptr<candidate::Hand>>
GraspDetector::pruneGraspCandidates(
    const util::Cloud &cloud,
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    double min_score) {
  // 1. Create grasp descriptors (images).
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list, images, hands);

  // 2. Classify the grasp candidates.
  std::vector<float> scores = classifier_->classifyImages(images);
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  // 3. Only keep grasps with a score larger than <min_score>.
  for (int i = 0; i < hands.size(); i++) {
    if (scores[i] > min_score) {
      hands[i]->setScore(scores[i]);
      hands_out.push_back(std::move(hands[i]));
    }
  }

  return hands_out;
}

void GraspDetector::printStdVector(const std::vector<int> &v,
                                   const std::string &name) const {
  printf("%s: ", name.c_str());
  for (int i = 0; i < v.size(); i++) {
    printf("%d ", v[i]);
  }
  printf("\n");
}

void GraspDetector::printStdVector(const std::vector<double> &v,
                                   const std::string &name) const {
  printf("%s: ", name.c_str());
  for (int i = 0; i < v.size(); i++) {
    printf("%3.2f ", v[i]);
  }
  printf("\n");
}

}  // namespace gpd
