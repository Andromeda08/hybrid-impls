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

struct GPUCameraData
{
    float4x4 view;
    float4x4 proj;
    float4x4 viewInverse;
    float4x4 projInverse;
    float4   eye;
    float    nearPlane;
    float    farPlane;

    float4x4 getViewProj()
    {
        return mul(proj, view);
    }

    // Get the linear depth (view Z) for a depth value in [0, 1] for a perspective projection.
    float getLinearDepth(float depth)
    {
        return (nearPlane * farPlane) / (farPlane - depth * (farPlane - nearPlane));
    }

    // View to World Space conversion for points.
    float3 viewToWorld_Point(float3 viewPosition)
    {
        return mul(viewInverse, float4(viewPosition, 1.0)).xyz;
    }

    // View to World Space conversion for vectors.
    float3 viewToWorld_Vector(float3 viewDirection)
    {
        return mul(float3x3(viewInverse), viewDirection);
    }

    // World to View Space conversion for points.
    float3 worldToView_Point(float3 worldPosition)
    {
        return mul(view, float4(worldPosition, 1.0)).xyz;
    }

    // World to View Space conversion for vectors.
    float3 worldToView_Vector(float3 worldDirection)
    {
        return mul(float3x3(view), worldDirection);
    }
};

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

struct ScreenSpaceTriangle
{
    float3 v0, v1, v2;
    float3 w0, w1, w2;
    float3 tangent;
    float3 color;
    uint   primitiveId;
};
