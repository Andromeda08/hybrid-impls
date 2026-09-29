#include "CyLoader.hpp"

#include <bitset>
#include <stdexcept>
#include <cyHairFile.h>
#include <fmt/format.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

HairGeometry CyLoader::loadModel(const std::filesystem::path& path)
{
    cyHairFile hairFile;

    if (const int32_t result = hairFile.LoadFromFile(path.string().c_str()); result == CY_HAIR_FILE_ERROR_CANT_OPEN_FILE)
    {
        throw std::runtime_error(fmt::format("Failed to open hair file: {}", path.string().c_str()));
    }

    HairGeometry g {};

    // Bit array in file header (https://www.cemyuksel.com/research/hairmodels/)
    const std::bitset<32> bits       = hairFile.GetHeader().arrays;
    const uint32_t        pointCount = hairFile.GetHeader().point_count;
    const uint32_t        hairCount  = hairFile.GetHeader().hair_count;

    // Process vertices
    {
        const float* points = hairFile.GetPointsArray();
        for (uint32_t i = 0; i < pointCount * 3; i += 3)
        {
            g.vertices.push_back({
                .position = glm::make_vec3(&points[i]),
            });
        }
    }

    // Process attributes
    {
        const bool      hasThicknessArray    = bits[2];
        const bool      hasTransparencyArray = bits[3];
        const bool      hasColorArray        = bits[4];
        const glm::vec3 defaultColor         = glm::make_vec3(hairFile.GetHeader().d_color);

        for (uint32_t i = 0, colorIdx = 0; i < pointCount; i++, colorIdx += 3)
        {
            g.attributes.push_back({
                .color        = hasColorArray ? glm::make_vec3(&hairFile.GetColorsArray()[colorIdx]) : defaultColor,
                .thickness    = hasThicknessArray ? hairFile.GetThicknessArray()[i] : hairFile.GetHeader().d_thickness,
                .transparency = hasTransparencyArray ? hairFile.GetTransparencyArray()[i] : hairFile.GetHeader().d_transparency,
            });
        }
    }

    // Process strands, strandlet breakdown
    {
        // Hair format stores segment count to get vertices: n+1
        const bool hasSegmentsArray = bits[0];
        if (hasSegmentsArray)
        {
            const uint16_t* segmentsArray = hairFile.GetSegmentsArray();
            for (auto i = 0; i < hairCount; i++)
            {
                g.strandVertexCounts.push_back(segmentsArray[i] + 1);
            }
        }

        std::span vertexSpan  = { g.vertices };
        uint32_t  firstVertex = 0;
        for (int32_t i = 0; i < hairCount; i++)
        {
            const auto vertexCount = hasSegmentsArray ? g.strandVertexCounts[i] : hairFile.GetHeader().d_segments + 1;
            if (!hasSegmentsArray)
            {
                g.strandVertexCounts.push_back(vertexCount);
            }

            // Process strand
            g.strands.push_back({
                .id          = i,
                .vertexCount = vertexCount,
                .vertices    = vertexSpan.subspan(firstVertex, vertexCount),
            });
            const auto& strand = g.strands.back();

            // Process strandlets
            const uint32_t strandletCount = (vertexCount > 1)
                ? ((vertexCount - 1) + (gHairMaxStrandletSize - 1) - 1) / (gHairMaxStrandletSize - 1)
                : 1;

            for (uint32_t j = 0; j < strandletCount; j++)
            {
                const uint32_t firstStrandletVertex = j * (gHairMaxStrandletSize - 1);  // stride (size - 1)
                const uint32_t remaining            = vertexCount - firstStrandletVertex;
                const uint32_t strandletVertexCount = std::min(gHairMaxStrandletSize, remaining);

                g.strandlets.push_back({
                    .strandId    = strand.id,
                    .vertexCount = strandletVertexCount,
                    .vertices    = strand.vertices.subspan(firstStrandletVertex, strandletVertexCount),
                });
            }

            // Strand description
            g.strandDescriptions.push_back({
                .strandId       = strand.id,
                .vertexCount    = vertexCount,
                .strandletCount = strandletCount,
                .firstVertex    = firstVertex,
            });

            firstVertex += vertexCount;
        }
    }

    g.taskGroupSizeX = (g.getStrandCount() + gHairMaxStrandletSize - 1) / gHairMaxStrandletSize;
    g.name           = path.filename().stem().string();

    return g;
}
