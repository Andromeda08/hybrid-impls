#pragma once

// Code for loading cyHairFiles.
// https://www.cemyuksel.com/research/hairmodels/
// ============================================================

#include <filesystem>
#include "HairGeometry.hpp"

struct CyLoader
{
    [[nodiscard]] static HairGeometry loadModel(const std::filesystem::path& path);
};