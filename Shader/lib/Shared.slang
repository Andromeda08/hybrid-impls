#pragma once

static const uint nbl_Hair_Workgroup_Size = 32;
static const uint nbl_StrandletStride     = nbl_Hair_Workgroup_Size - 1;

// PCG
uint hash(uint x)
{
    uint state = x * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

#include "Camera.slang"

struct TaskPayload
{
    // Global strand index of the first strand in the [Task] workgroup.
    uint baseID;

    // Exclusive prefix sum of strandlet counts within this [Task] workgroup.
    uint8_t deltaID[nbl_Hair_Workgroup_Size];
};

struct MeshVertex
{
    float4 position : SV_Position;
    float3 worldPosition;
    float3 worldTangent;
    float3 color;
};

struct HairVertex
{
    float3 position;
};

struct HairAttributes
{
    float3 color;
    float thickness;
    float transparency;
};

struct HairStrandDesc
{
    int strandId;
    uint vertexCount;
    uint strandletCount;
    uint firstVertex;
};

enum ColorMode : uint
{
    Model     = 0,
    Strand    = 1,
    Strandlet = 2,
    Quad      = 3,
    Compute   = 4,
}

static const uint nbl_VizColorCount = 12;
static const float3 colorPool[nbl_VizColorCount] = {
    float3(234, 118, 203), float3(136,  57, 239), float3(210,  15,  57), float3(230,  69,  83),
    float3(254, 100,  11), float3(221, 120, 120), float3(223, 142,  29), float3( 64, 160,  43),
    float3( 23, 146, 153), float3( 32, 159, 181), float3( 30, 102, 245), float3(144, 135, 253),
};

float3 getColor(uint i)
{
    return colorPool[i % nbl_VizColorCount] / 255.0;
};

uint getStrandletVertexCount(uint strandletID, uint totalVertexCount)
{
    return min(totalVertexCount - (nbl_StrandletStride * strandletID), nbl_Hair_Workgroup_Size);
}

bool isSmallTriangle(float2 a, float2 b, float2 c, float threshold)
{
    float2 bbMin = min(min(a, b), c);
    float2 bbMax = max(max(a, b), c);
    float2 size  = bbMax - bbMin;
    return max(size.x, size.y) < threshold;
}

float2 toScreen(float4 clipPos, float2 viewportSize)
{
    float2 ndc = clipPos.xy / clipPos.w;
    return (ndc * 0.5 + 0.5) * viewportSize;
}

// Given the mesh workgroup id, resolve the strand and strandlet the current thread is working on.
void resolveMeshTask(
        uint    groupId,
        uint    baseId,
        uint8_t deltaIds[nbl_Hair_Workgroup_Size],
    out uint    strandId,
    out uint    strandletId
)
{
    uint k       = 0;   // Strand ID delta
    uint deltaId = 0;   // Strandlet ID delta
    for (uint i = 0; i < nbl_Hair_Workgroup_Size; i++)
    {
        if (groupId < uint(deltaIds[i]))
        {
            break;
        }
        deltaId = uint(deltaIds[i]);
        k       = i;
    }

    strandId    = baseId + k;
    strandletId = groupId - deltaId;
}

float3 kajiyaKay(float3 diffuse, float3 specular, float p, float3 T, float3 L, float3 V)
{
    float cosTL = dot(T, L);
    float sinTL = sqrt(max(0.0, 1.0 - cosTL * cosTL));

    float3 H = normalize(L + V);
    float cosTH = dot(T, H);
    float sinTH = sqrt(max(0.0, 1.0 - cosTH * cosTH));

    float3 d = diffuse * sinTL;
    float3 s = specular * pow(max(sinTH, 0.0), p);

    return d + s;
}

float getSignedArea(float2 a, float2 b, float2 c)
{
    float2 AB = float2(b.x - a.x, b.y - a.y);
    float2 AC = float2(c.x - a.x, c.y - a.y);
    return 0.5 * determinant(float2x2(AB, AC));
}

struct ScreenSpaceTriangle
{
    float3 v0, v1, v2;
    float3 tangent;
    float3 color;
    uint   primitiveId;
};

void writeTriangle(
    float3 sd0, float3 sd1, float3 sd2, float3 tangent, float3 color, uint primId,
    uint maxCount, ScreenSpaceTriangle* pTarget, uint slot
)
{
    if (slot >= maxCount) { return; }

    ScreenSpaceTriangle tri;
    tri.v0 = sd0; tri.v1 = sd1; tri.v2 = sd2;
    tri.tangent = tangent;
    tri.color = color;
    tri.primitiveId = primId;

    pTarget[slot] = tri;
}

void appendTriangle(
    float3 sd0, float3 sd1, float3 sd2, float3 tangent, float3 color, uint primId,
    uint maxCount, uint* pCounter, ScreenSpaceTriangle* pTarget
)
{
    uint slot;
    InterlockedAdd(*pCounter, 1, slot);
    writeTriangle(sd0, sd1, sd2, tangent, color, primId, maxCount, pTarget, slot);
}

struct BBoxInt2
{
    int2 bboxMin;
    int2 bboxMax;

    bool isEmpty()
    {
        return any(bboxMin > bboxMax);
    }

    static BBoxInt2 fromTriangle(float2 p0, float2 p1, float2 p2)
    {
        float2 _min = min(min(p0, p1), p2);
        float2 _max = max(max(p0, p1), p2);

        BBoxInt2 result;
        result.bboxMin = int2(floor(_min - 0.5)) + 1;
        result.bboxMax = int2(floor(_max - 0.5));
        return result;
    }

    static BBoxInt2 fromScissor(uint xmin, uint ymin, uint xmax, uint ymax)
    {
        BBoxInt2 result;
        result.bboxMin = int2(xmin, ymin);
        result.bboxMax = int2(xmax, ymax);
        return result;
    }

    BBoxInt2 getClippedBy(BBoxInt2 other)
    {
        BBoxInt2 result;
        result.bboxMin = max(bboxMin, other.bboxMin);
        result.bboxMax = min(bboxMax, other.bboxMax);
        return result;
    }
};

struct Triangle
{
    float2   p0, p1, p2;

    BBoxInt2 bounds;
    float    signedArea;
    float    invArea;
    float    s;

    static Triangle fromInput(ScreenSpaceTriangle t)
    {
        Triangle tri;

        tri.p0 = t.v0.xy;
        tri.p1 = t.v1.xy;
        tri.p2 = t.v2.xy;

        tri.bounds     = BBoxInt2::fromTriangle(tri.p0, tri.p1, tri.p2);
        tri.signedArea = getSignedArea(tri.p0, tri.p1, tri.p2);
        tri.invArea    = 1.0f / tri.signedArea;
        tri.s          = sign(tri.signedArea);

        return tri;
    }

    bool isDegenerate()
    {
        return abs(signedArea) < 0.0000005;
    }

    float3 bary(float2 p)
    {
        float w0 = getSignedArea(p1, p2, p);
        float w1 = getSignedArea(p2, p0, p);
        float w2 = getSignedArea(p0, p1, p);
        return float3(w0, w1, w2) * invArea;
    }
};
