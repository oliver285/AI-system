#include "../include/image_processor.h"

image_processor::image_processor() = default;
image_processor::~image_processor() = default;
cv::Mat image_processor::load_image(const std::string& image_path) {
    // Load image in grayscale
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    return img;
}


cv::Mat image_processor::preprocess_image(const cv::Mat& img) {
    cv::Mat resized_img;
    // Resize to 28x28 (MNIST standard)
    cv::resize(img, resized_img, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    // Normalize pixel values to [0, 1] (optional)
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
    return resized_img;
}

/* Dirct Memory access*/
// Matrix image_processor::flatten_image(const cv::Mat& img) {
//        if (img.empty()) {
//         throw std::invalid_argument("Input image is empty");
//     }
//    else if (img.rows != IMG_HEIGHT || img.cols != IMG_WIDTH) {
//         throw std::invalid_argument("Input image dimensions must be " + 
//                                   std::to_string(IMG_WIDTH) + "x" + 
//                                   std::to_string(IMG_HEIGHT));
//     }

//     // Convert to float if needed and normalize to [0,1]
//     cv::Mat processed_img;
//     if (img.type() != CV_32F) {
//         img.convertTo(processed_img, CV_32F, 1.0/255.0);
//     } else {
//         processed_img = img.clone();
//     }
    
//     cv::Mat flat_img = processed_img.reshape(1, 1);  // Single row
//     Matrix flattened(flat_img.cols, 1);  // Column vector
    
//     // Direct memory copy (fast but assumes matrix is continuous)
//     if (flat_img.isContinuous()) {
//         std::memcpy(flattened.data.data(), 
//                    flat_img.ptr<float>(0), 
//                    flat_img.cols * sizeof(float));
//     } else {
//         // Fallback to safe iteration if not continuous
//         float* ptr = flat_img.ptr<float>(0);
//         for (size_t i = 0; i < flattened.size(); ++i) {
//             flattened.data[i] = ptr[i];
//         }
//     }
    
//     return flattened;
// }

/*Non direct memory access*/
Matrix image_processor::flatten_image(const cv::Mat& img) {
    // Validate input image
    if (img.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    if (img.rows != IMG_HEIGHT || img.cols != IMG_WIDTH) {
        throw std::invalid_argument("Input image dimensions must be " + 
                                  std::to_string(IMG_WIDTH) + "x" + 
                                  std::to_string(IMG_HEIGHT));
    }

    // Convert to float32 and normalize to [0,1] in one step
    cv::Mat processed_img;
    img.convertTo(processed_img, CV_32F, 1.0/255.0);

    // Reshape to column vector (rows = height*width, cols = 1)
    cv::Mat flat_img = processed_img.reshape(1, IMG_HEIGHT * IMG_WIDTH);

    // Initialize output matrix
    Matrix flattened(flat_img.rows, flat_img.cols);

    // Efficient data transfer using pointer arithmetic
    const float* src_ptr = flat_img.ptr<float>(0);
    for (size_t i = 0; i < flattened.row_count(); ++i) {
        for (size_t j = 0; j < flattened.col_count(); ++j) {
            // Calculate linear index (since we know it's a column vector)
            flattened(i, j) = src_ptr[i * flattened.col_count() + j];
        }
    }

    return flattened;
}


// Load only images (without labels)
Matrix image_processor::load_images(const std::string& folder_path) {
    std::vector<std::filesystem::path> valid_files = get_valid_image_files(folder_path);
    Matrix images(valid_files.size(), IMG_WIDTH * IMG_HEIGHT);
    
    // Convert paths to strings
    // std::vector<std::string> file_paths;
    // file_paths.reserve(valid_files.size()); // Pre-allocate memory
    int count=0;
    for (const auto& path : valid_files) {
        // file_paths.push_back(path.string()); // Convert each path to string
        load_image_data(path.string(), images,count);
        count++;
    }
    
    // load_image_data(file_paths, images); // Now pass the vector of strings
    return images;
}


Matrix image_processor::load_dataset(const std::string& folder_path){

   
    
    return load_images(folder_path);
   

}

// Load only labels
Matrix image_processor::load_labels(const std::string& folder_path, int label) {
    std::vector<std::filesystem::path> valid_files = get_valid_image_files(folder_path);
    Matrix labels(valid_files.size(), 1);
    
    for (size_t i = 0; i < valid_files.size(); ++i) {
        labels(i, 0) = static_cast<float>(label);
    }
    return labels;
}

// Helper function to get valid image files
std::vector<std::filesystem::path> image_processor::get_valid_image_files(
    const std::string& folder_path) {
    std::vector<std::filesystem::path> valid_files;
    
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            const auto ext = entry.path().extension();
            if (ext == ".jpg" || ext == ".png") {
                valid_files.push_back(entry.path());
            }
        }
    }
    
    if (valid_files.empty()) {
        throw std::runtime_error("No valid image files in: " + folder_path);
    }
    
    return valid_files;
}

// Helper function to load image data
void image_processor::load_image_data(
    // std::vector<std::filesystem::path> valid_files = get_valid_image_files(folder_path);
    // Matrix images(valid_files.size(), IMG_WIDTH * IMG_HEIGHT);
    // // Convert paths to strings
    // std::vector<std::string> file_paths;
    // file_paths.reserve(valid_files.size()); // Pre-allocate memory


    const std::string folder_path,Matrix& output,int i) {
        
 
        try {

            cv::Mat img = load_image(folder_path);
             // const std::vector<std::string>& files
    // Matrix img_data = load_images(folder_path);
            cv::Mat processed_img = preprocess_image(img);
            Matrix flattened = flatten_image(processed_img);
            
            
            for (size_t col = 0; col < flattened.size(); ++col) { 
                output(i, col) = flattened.no_bounds_check(col);
            
        } 
    }catch (const std::exception& e) {
            std::cerr << "Error processing " << folder_path << ": " << e.what() << '\n';
            // Fill with zeros if error occurs
            for (size_t col = 0; col < output.col_count(); ++col) {
                output(i, col) = 0.0f;
            }
        
    }
}



void image_processor::save_to_csv(const Matrix& dataset, const std::string& csv_path) {
    // Open file with error checking
    std::ofstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + csv_path);
    }

    // Set precision for floating-point values
    file << std::fixed << std::setprecision(6);

    // Write data
    for (size_t i = 0; i < dataset.row_count(); ++i) {
        for (size_t j = 0; j < dataset.col_count(); ++j) {
            file << dataset(i, j);
            // Only add comma if not last column
            if (j < dataset.col_count() - 1) {
                file << ",";
            }
        }
        // No newline after last row
        if (i < dataset.row_count() - 1) {
            file << "\n";
        }
    }

    // Explicit close (though destructor would do this)
    file.close();
}

void image_processor::shuffle_dataset(Matrix& images, Matrix& labels) {
    if (images.row_count() != labels.row_count()) {
        throw std::invalid_argument("Images and labels must have same number of rows");
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    
    for (size_t i = images.row_count() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);
        
        // Swap rows in both matrices
        for (size_t col = 0; col < images.col_count(); ++col) {
            std::swap(images(i, col), images(j, col));
        }
        std::swap(labels(i, 0), labels(j, 0));
    }
}




int main() {
    image_processor processor;

    try {
        // Load datasets
        auto cracked_images = processor.load_dataset("../datasets/testing/data/Cracked");
        auto cracked_labels = processor.load_labels("../datasets/testing/data/Cracked", 1);
        auto noncracked_images = processor.load_dataset("../datasets/testing/data/NonCracked");
        auto  noncracked_labels = processor.load_labels("../datasets/testing/data/NonCracked", 0);
        uint8_t IMG_WIDTH=28, IMG_HEIGHT =28;
        // Combine images and labels
        Matrix all_images(cracked_images.row_count() + noncracked_images.row_count(), 
                         IMG_WIDTH * IMG_HEIGHT);
        Matrix all_labels(cracked_labels.row_count() + noncracked_labels.row_count(), 1);

        // Copy cracked data
        for (size_t i = 0; i < cracked_images.row_count(); ++i) {
            for (size_t j = 0; j < cracked_images.col_count(); ++j) {
                all_images(i, j) = cracked_images(i, j);
            }
            all_labels(i, 0) = cracked_labels(i, 0);
        }

        // Copy non-cracked data
        for (size_t i = 0; i < noncracked_images.row_count(); ++i) {
            for (size_t j = 0; j < noncracked_images.col_count(); ++j) {
                all_images(i + cracked_images.row_count(), j) = noncracked_images(i, j);
            }
            all_labels(i + cracked_labels.row_count(), 0) = noncracked_labels(i, 0);
        }

        // Shuffle the combined dataset
        processor.shuffle_dataset(all_images, all_labels);

        // Save to CSV (you might want to save images and labels separately)
        processor.save_to_csv(all_images, "crack_images.csv");
        processor.save_to_csv(all_labels, "crack_labels.csv");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
  

