#pragma once

#include <cstdint>
#include <span>
#include <glm/glm.hpp>

using StrandId = int32_t;

/**
 * Maximum strandlet size.
 */
constexpr uint32_t gHairMaxStrandletSize = 32;

/**
 * Hair Vertex Data Representation [CPU, GPU]
 */
struct HairVertex
{
    glm::vec3 position;
};

/**
 * Hair Vertex Attribute Data [CPU, GPU]
 */
struct HairAttributes
{
    glm::vec3 color;        // RGB
    float     thickness;    // Thickness of the strand at this position
    float     transparency; // Transparency at this point
};

/**
 * Hair Strand Data [CPU]
 * A hair strand consists of several points that are later broken down into strandlets.
 */
struct HairStrand
{
    StrandId              id          = 0;  // Referenced by Strandlets
    uint32_t              vertexCount = 0;  // No. vertices in the strand
    std::span<HairVertex> vertices    = {}; // Non-owning view of the vertices of the strand
};

/**
 * Hair Strandlet Data [CPU]
 * A hair strandlet is a segment of a maximum fixed size of a hair strand.
 */
struct HairStrandlet
{
    StrandId              strandId    = 0;  // Which strand this strandlet belongs to
    uint32_t              vertexCount = 0;  // No. vertices in the strandlet
    std::span<HairVertex> vertices    = {}; // Non-owning view of the vertices of the strandlet
};

/**
 * Struct describing a single hair strand [CPU, GPU]
 */
struct HairStrandDesc
{
    StrandId strandId       = 0;
    uint32_t vertexCount    = 0;
    uint32_t strandletCount = 0;
    uint32_t firstVertex    = 0;
};

/**
 * Struct for passing buffer addresses to the GPU [CPU, GPU]
 */
struct HairBufferAddresses
{
    uint64_t vertexBuffer     = 0;
    uint64_t strandDescBuffer = 0;
};