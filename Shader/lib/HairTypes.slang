#pragma once

#include "Constants.slang"

struct HairVertex
{
    float3 position;
};

struct HairAttributes
{
    float3 color;
    float  thickness;
    float  transparency;
};

struct HairStrandDescription
{
    int  strandID;
    uint vertexCount;
    uint strandletCount;
    uint firstVertex;

    uint getStrandletFirstVertex(uint strandletID)
    {
        return firstVertex + strandletID * cfg::StrandletStride;
    }

    uint getStrandletVertexCount(uint strandletID)
    {
        return min(
            vertexCount - (cfg::StrandletStride * strandletID),
            cfg::WorkgroupSize
        );
    }
};
