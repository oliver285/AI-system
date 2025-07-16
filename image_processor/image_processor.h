#include "../ML/matrix.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <string>
// #include <random>
#include <chrono>
// #include <algorithm>
class image_processor {
    public:
        // Add these declarations in your class definition
        Matrix load_images(const std::string& folder_path);
        Matrix load_labels(const std::string& folder_path, int label);
        uint8_t IMG_WIDTH=28, IMG_HEIGHT =28;
        image_processor();
        ~image_processor();
    private:
        // Helper functions
        std::vector<std::filesystem::path> get_valid_image_files(const std::string& folder_path);
        void load_image_data(const std::vector<std::string>& files, Matrix& output);
        // Constructor/Destructor
 
    
        // Core Methods
        cv::Mat load_image(const std::string& image_path);
        cv::Mat preprocess_image(const cv::Mat& img);
        Matrix flatten_image(const cv::Mat& img);
        Matrix load_dataset(const std::string& folder_path, int label);
    
        // Utility Methods
        void save_to_csv(const Matrix& dataset, const std::string& csv_path);
        void shuffle_dataset(Matrix& dataset);
    };

// def predict_new_image(image_path, W1, b1, W2, b2):
//     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
//     img = cv2.resize(img, (28, 28))
//     img_flatten = img.flatten().reshape(-1, 1)
//     _, _, _, A2 = forward_prop(W1, b1, W2, b2, img_flatten)
//     prediction = get_predictions(A2)[0]
//     label = "Cracked" if prediction == 1 else "Not Cracked"
//     print(f"{image_path}: {label}")
//     return label

// # =======================
// # Preprocessing Dataset
// # =======================
// def load_images_from_folder(folder, label, image_size=(28, 28)):
//     data = []
//     for filename in os.listdir(folder):
//         img_path = os.path.join(folder, filename)
//         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
//         if img is not None:
//             img = cv2.resize(img, image_size)
//             img_flatten = img.flatten()
//             data.append(np.insert(img_flatten, 0, label))
//     return data

// cracked = load_images_from_folder("datasets/testing/data/Cracked", label=1)
// not_cracked = load_images_from_folder("datasets/testing/data/NonCracked", label=0)
// all_data = np.array(cracked + not_cracked)
// np.random.shuffle(all_data)
// pd.DataFrame(all_data).to_csv("crack_dataset.csv", index=False)
