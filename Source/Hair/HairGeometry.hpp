#pragma once

#include <string>
#include <vector>
#include "HairTypes.hpp"

struct HairGeometry
{
    /**
     * Vertices belong 1-1 to the attributes at the same index, therefore there's an equal number of both.
     */
    [[nodiscard]] uint32_t getVertexCount() const noexcept { return vertices.size(); }

    [[nodiscard]] uint32_t getStrandCount() const noexcept { return strandDescriptions.size(); }

    std::vector<HairVertex>     vertices;
    std::vector<HairAttributes> attributes;
    std::vector<HairStrandDesc> strandDescriptions;

    std::vector<uint32_t>       strandVertexCounts;
    std::vector<HairStrand>     strands;
    std::vector<HairStrandlet>  strandlets;

    uint32_t                    taskGroupSizeX;
    std::string                 name;
};

/**
 * Stores the offsets and data count into the global buffers for each hair model.
 * (hair vertex, attribute and strand descriptions)
 */
struct GlobalHairInfo
{
    uint32_t firstVertex;
    uint32_t vertexCount;

    uint32_t firstAttribute;
    uint32_t attributeCount;

    uint32_t firstStrand;
    uint32_t strandCount;
};
