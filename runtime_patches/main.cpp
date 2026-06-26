/*
 * JRDB PointPillar v2 runtime entry (patched from NVIDIA CUDA-PointPillars main.cpp)
 */
#include <cuda_runtime.h>

#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>

#include "pointpillar.hpp"
#include "common/check.hpp"

void GetDeviceInfo(void)
{
  cudaDeviceProp prop;
  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
  }
  printf("\n");
}

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    }
    return false;
}

int getFolderFile(const char *path, std::vector<std::string>& files, const char *suffix = ".bin")
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path)) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string file = ent->d_name;
            if (hasEnding(file, suffix)) {
                files.push_back(file.substr(0, file.length() - 4));
            }
        }
        closedir(dir);
    } else {
        printf("No such folder: %s.", path);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);
    if (!dataFile.is_open()) {
        std::cout << "Can't open files: " << file << std::endl;
        return -1;
    }
    dataFile.seekg(0, dataFile.end);
    unsigned int len = dataFile.tellg();
    dataFile.seekg(0, dataFile.beg);
    char *buffer = new char[len];
    dataFile.read(buffer, len);
    dataFile.close();
    *data = (void *)buffer;
    *length = len;
    return 0;
}

void SaveBoxPred(std::vector<pointpillar::lidar::BoundingBox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " " << box.y << " " << box.z << " "
              << box.w << " " << box.l << " " << box.h << " "
              << box.rt << " " << box.id << " " << box.score << "\n";
        }
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
}

std::shared_ptr<pointpillar::lidar::Core> create_core() {
    pointpillar::lidar::VoxelizationParameter vp;
    // JRDB pointpillar_v2 / jrdb_v2_bs8
    vp.min_range = nvtype::Float3(-25.6f, -25.6f, -2.0f);
    vp.max_range = nvtype::Float3(25.6f, 25.6f, 4.0f);
    vp.voxel_size = nvtype::Float3(0.128f, 0.128f, 6.0f);
    vp.grid_size = vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
    vp.max_voxels = 50000;
    vp.max_points_per_voxel = 32;
    vp.max_points = 300000;
    vp.num_feature = 4;

    pointpillar::lidar::PostProcessParameter pp;
    pp.min_range = vp.min_range;
    pp.max_range = vp.max_range;
    pp.feature_size = nvtype::Int2(vp.grid_size.x / 2, vp.grid_size.y / 2);
    pp.num_classes = 1;
    pp.num_anchors = 4;
    pp.num_dir_bins = 4;
    pp.len_per_anchor = 4;
    pp.anchors[0] = 0.93f; pp.anchors[1] = 0.51f; pp.anchors[2] = 1.89f; pp.anchors[3] = 0.0f;
    pp.anchors[4] = 0.93f; pp.anchors[5] = 0.51f; pp.anchors[6] = 1.89f; pp.anchors[7] = 0.78539816f;
    pp.anchors[8] = 0.93f; pp.anchors[9] = 0.51f; pp.anchors[10] = 1.89f; pp.anchors[11] = 1.57079633f;
    pp.anchors[12] = 0.93f; pp.anchors[13] = 0.51f; pp.anchors[14] = 1.89f; pp.anchors[15] = 2.35619449f;
    pp.anchor_bottom_heights = nvtype::Float3(-0.95f, 0.0f, 0.0f);
    pp.score_thresh = 0.25f;
    pp.dir_offset = 0.78539f;
    pp.nms_thresh = 0.15f;

    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp;
    param.lidar_model = "../../model/pointpillar_v2.plan";
    param.lidar_post = pp;
    return pointpillar::lidar::create_core(param);
}

static bool startswith(const char *s, const char *with, const char **last)
{
    while (*s++ == *with++) {
        if (*s == 0 || *with == 0) break;
    }
    if (*with == 0) *last = s + 1;
    return *with == 0;
}

static void help()
{
    printf(
        "Usage:\n"
        "    ./pointpillar in/ out/ [--timer]\n"
        "    Run JRDB PointPillar v2 TRT inference on .bin point clouds\n"
    );
    exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) help();

    const char *in_dir = argv[1];
    const char *out_dir = argv[2];
    const char *value = nullptr;
    bool timer = false;
    if (argc == 4 && startswith(argv[3], "--timer", &value)) {
        timer = true;
    }

    GetDeviceInfo();

    std::vector<std::string> files;
    getFolderFile(in_dir, files);
    std::cout << "Total " << files.size() << std::endl;

    auto core = create_core();
    if (core == nullptr) {
        printf("Core init failed.\n");
        return -1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    core->print();
    core->set_timer(timer);

    for (const auto &file : files) {
        std::string dataFile = std::string(in_dir) + file + ".bin";
        unsigned int length = 0;
        void *data = NULL;
        loadData(dataFile.data(), &data, &length);
        std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
        int points_size = length / sizeof(float) / 4;

        auto bboxes = core->forward((float *)buffer.get(), points_size, stream);
        std::string save_file_name = std::string(out_dir) + file + ".txt";
        SaveBoxPred(bboxes, save_file_name);
    }

    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}
