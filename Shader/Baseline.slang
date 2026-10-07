#language slang 2026

#include "lib/Shared.slang"

struct PushConstants
{
    float4x4        model;
    HairVertex*     vertices;
    HairAttributes* attributes;
    HairStrandDesc* strandDescriptions;
    GPUCameraData*  camera;
    uint            strandCount;

    HairStrandDesc getStrandDescription(uint globalStrandIndex)
    {
        // A single buffer is used for all hair models, we need the global index for that buffer.
        return strandDescriptions[globalStrandIndex];
    }
};
[[vk::push_constant]] PushConstants pc;

groupshared TaskPayload taskOut;

[shader("amplification")]
[numthreads(nbl_Hair_Workgroup_Size, 1, 1)]
[require(spvGroupNonUniformArithmetic, spvGroupNonUniformShuffle)]
void taskMain(
    uint  threadIndex : SV_GroupIndex,
    uint3 groupID     : SV_GroupID
)
{
    const uint baseID = groupID.x * nbl_Hair_Workgroup_Size;

    const uint localStrandIndex  = threadIndex;
    const uint globalStrandIndex = baseID + localStrandIndex;

    uint strandletCount = (globalStrandIndex < pc.strandCount)
        ? pc.getStrandDescription(globalStrandIndex).strandletCount
        : 0;

    uint strandWorkgroupOffset = WavePrefixSum(strandletCount);

    taskOut.deltaID[threadIndex] = uint8_t(strandWorkgroupOffset);
    taskOut.baseID = baseID;

    GroupMemoryBarrierWithGroupSync();

    uint sumStrandletCount = WaveReadLaneAt(strandWorkgroupOffset + strandletCount, nbl_Hair_Workgroup_Size - 1);
    DispatchMesh(sumStrandletCount, 1, 1, taskOut);
}

uint getStrandletVertexCount(uint strandletID, uint totalVertexCount)
{
    return min(totalVertexCount - (nbl_StrandletStride * strandletID), nbl_Hair_Workgroup_Size);
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

[shader("mesh")]
[numthreads(nbl_Hair_Workgroup_Size, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    // Workgroup
    uint  laneId   : SV_GroupIndex,
    uint3 groupId  : SV_GroupID,
    // Mesh Shader
    in  payload  TaskPayload inPayload,
    out vertices MeshVertex  verts[64],
    out indices  uint3       tris[64]
)
{
    uint strandId, strandletId;
    resolveMeshTask(groupId.x, inPayload.baseID, inPayload.deltaID, strandId, strandletId);

    // Thread [Strand] information
    HairStrandDesc  strandDesc = pc.getStrandDescription(strandId);

    // Thread [Strandlet] information
    uint strandletFirstVertex = strandletId * nbl_StrandletStride;
    uint strandletVertices    = getStrandletVertexCount(strandletId, strandDesc.vertexCount);

    // Thread specific offset into hair vertex buffers computed from strand & strandlet
    uint vertexBufferOffset = strandDesc.firstVertex + strandletFirstVertex + laneId;

    // Calculate output parameters
    uint quadCount = strandletVertices - 1;
    SetMeshOutputCounts(strandletVertices * 2, quadCount * 2); // (Vtx, Prim)

    // Early return for shorter than wg size strandlets
    if (laneId >= strandletVertices)
    {
        return;
    }

    float4 strandVertex = float4(pc.vertices[vertexBufferOffset].position, 1.0);

    // Tangent is computed from the previous vertex at the last vertex of a strand.
    bool isStrandEnd = strandletFirstVertex + laneId == strandDesc.vertexCount - 1;
    float3 T = normalize(isStrandEnd
        ? pc.vertices[vertexBufferOffset - 1].position - strandVertex.xyz
        : strandVertex.xyz - pc.vertices[vertexBufferOffset + 1].position);

    // Camera facing quad extrusion
    float4 wStrandPos   = mul(pc.model, strandVertex);
    float3 worldTangent = normalize(mul((float3x3) pc.model, T));

    float3 V = normalize(pc.camera.eye.xyz - wStrandPos.xyz);
    float4 offsetVector = float4(normalize(cross(V, worldTangent)), 0.0);
    float4 wOffsetPos = wStrandPos + offsetVector * pc.attributes[vertexBufferOffset].thickness;

    float4 A = (laneId % 2 == 0) ? wStrandPos : wOffsetPos;
    float4 B = (laneId % 2 == 0) ? wOffsetPos : wStrandPos;

    const var  VP        = pc.camera.getViewProj();
    const uint outOffset = laneId * 2;

    verts[outOffset + 0] = { mul(VP, A), A.xyz, worldTangent, pc.attributes[vertexBufferOffset].color };
    verts[outOffset + 1] = { mul(VP, B), B.xyz, worldTangent, pc.attributes[vertexBufferOffset].color };

    if (laneId < quadCount)
    {
        const uint triOffset = laneId * 2;
        tris[triOffset + 0] = uint3(2, 1, 0) + outOffset;
        tris[triOffset + 1] = uint3(3, 2, 0) + outOffset;
    }
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

[shader("fragment")]
float4 fragMain(MeshVertex input) : SV_Target
{
    const float3 lightPos = float3(-75.0, 125.0, 50.0);

    float3 T = normalize(input.worldTangent);
    float3 L = normalize(lightPos.xyz - input.worldPosition);
    float3 V = normalize(pc.camera.eye.xyz - input.worldPosition);

    float3 diffuse  = float3(0.32549, 0.23921, 0.20784);
    float3 specular = float3(0.41568, 0.30588, 0.21960);

    float3 color = kajiyaKay(diffuse, specular, 16.0, T, L, V);
    return float4(color, 1.0);
}
