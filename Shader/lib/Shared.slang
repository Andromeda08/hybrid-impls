#pragma once

static const uint nbl_Hair_Workgroup_Size = 32;

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
