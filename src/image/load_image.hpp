// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef LOAD_IMAGE_HPP
#define LOAD_IMAGE_HPP

#include <filesystem>
#include <set>
#include <vector>
#include <openvino/openvino.hpp>

namespace utils {
    std::vector<ov::Tensor> load_images(const std::filesystem::path& input_path);
    ov::Tensor load_image(const std::filesystem::path& image_path);
}

#endif // LOAD_IMAGE_HPP
