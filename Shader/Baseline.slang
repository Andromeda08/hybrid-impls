#language slang 2026

#include "lib/Shared.slang"

struct PushConstants
{
    float4x4        model;
    HairVertex*     vertices;
    HairAttributes* attributes;
    HairStrandDesc* strandDescriptions;
    GPUCameraData*  camera;

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

    uint strandletCount        = pc.getStrandDescription(globalStrandIndex).strandletCount;
    uint strandWorkgroupOffset = WavePrefixSum(strandletCount);

    if (threadIndex != 0)
    {
        taskOut.deltaID[threadIndex] = uint8_t(strandWorkgroupOffset);
    }
    taskOut.baseID = baseID;

    GroupMemoryBarrierWithGroupSync();

    uint sumStrandletCount = WaveReadLaneAt(strandWorkgroupOffset + strandletCount, nbl_Hair_Workgroup_Size - 1);
    DispatchMesh(sumStrandletCount, 1, 1, taskOut);
}

struct MeshData
{
    float4 position : SV_Position;
    float4 worldPosition;
    float4 worldTangent;
};

uint getStrandletVertexCount(uint strandletID, uint totalVertexCount)
{
    return min(totalVertexCount - (nbl_Hair_Workgroup_Size * strandletID), nbl_Hair_Workgroup_Size);
}

[shader("mesh")]
[numthreads(nbl_Hair_Workgroup_Size, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    // Workgroup
    uint  threadIndex   : SV_GroupIndex,
    uint3 groupID       : SV_GroupID,
    // Mesh Shader
    in  payload  TaskPayload inPayload,
    out vertices MeshData    verts[64],
    out indices  uint3       tris[64]
)
{
    uint workGroupID = groupID.x;
    uint laneID      = threadIndex;

    uint deltaID = 0;
    uint k       = 0;
    for (uint i = 0; i < nbl_Hair_Workgroup_Size; i++)
    {
        if (workGroupID < uint(inPayload.deltaID[i]))
        {
            break;
        }
        deltaID = uint(inPayload.deltaID[i]);
        k = i;
    }

    // Current [Strand] information
    uint            current_strandID    = inPayload.baseID + k;
    HairStrandDesc  strand_description  = pc.getStrandDescription(current_strandID);
    uint            strand_vertex_count = strand_description.vertexCount;
    uint            base_vertex_offset  = strand_description.firstVertex;

    // Current [Strandlet] information
    uint strandletID        = workGroupID - deltaID;
    uint strandlet_vertices = getStrandletVertexCount(strandletID, strand_vertex_count);

    // Calculate output parameters
    uint n_quads = strandlet_vertices - 1;
    uint n_tri   = n_quads * 2;
    uint n_vtx   = strandlet_vertices * 2;

    SetMeshOutputCounts(n_vtx, n_tri);

    // Do no work if current lane exceeds quad count
    if (laneID > n_quads)
    {
        return;
    }

    // Calculate global offset into the vertex buffer
    uint vertex_buffer_offset = base_vertex_offset + (strandletID * nbl_Hair_Workgroup_Size) + laneID - strandletID;

    float4 strand_vertex = float4(pc.vertices[vertex_buffer_offset].position, 1.0);
    float4 offset_vertex = strand_vertex + float4(pc.attributes[vertex_buffer_offset].thickness, 0.0, 0.0, 0.0);
    float4 next_vertex   = float4(pc.vertices[vertex_buffer_offset + 1].position, 1.0);

    float4 tangent = float4(strand_vertex.xyz - next_vertex.xyz, 0.0);

    float4 A = (laneID % 2 == 0) ? strand_vertex : offset_vertex;
    float4 B = (laneID % 2 == 0) ? offset_vertex : strand_vertex;

    var M  = pc.model;
    var VP = mul(pc.camera.proj, pc.camera.view);

    float4 world_pos_A   = mul(M, A);
    float4 world_pos_B   = mul(M, B);
    float4 world_tangent = normalize(float4(mul(M, tangent).xyz, 0.0));

    const uint out_offset = laneID * 2;

    verts[out_offset + 0].position      = mul(VP, world_pos_A);
    verts[out_offset + 0].worldPosition = world_pos_A;
    verts[out_offset + 0].worldTangent  = world_tangent;

    verts[out_offset + 1].position      = mul(VP, world_pos_B);
    verts[out_offset + 1].worldPosition = world_pos_B;
    verts[out_offset + 1].worldTangent  = world_tangent;

    const uint tri_offset = laneID * 2;
    tris[tri_offset + 0] = uint3(2, 1, 0) + out_offset;
    tris[tri_offset + 1] = uint3(3, 2, 0) + out_offset;
}

float3 kajiyaKay(float3 diffuse, float3 specular, float p, float3 T, float3 L, float3 V)
{
    float cosTL = dot(T, L);
    float sinTL = sqrt(max(0.0, 1.0 - cosTL * cosTL));

    float3 H = normalize(L + V);
    float cosTH = dot(T, H);
    float sinTH = sqrt(1.0 - cosTH * cosTH);

    float3 d = diffuse * sinTL;
    float3 s = specular * pow(max(sinTH, 0.0), p);

    return d + s;
}

[shader("fragment")]
float4 fragMain(MeshData input) : SV_Target
{
    const float3 lightPos = float3(-75.0, 125.0, 50.0);

    float3 T = normalize(input.worldTangent.xyz);
    float3 L = normalize(lightPos.xyz - input.worldPosition.xyz);
    float3 V = normalize(pc.camera.eye.xyz - input.worldPosition.xyz);

    float3 diffuse  = float3(0.32549, 0.23921, 0.20784);
    float3 specular = float3(0.41568, 0.30588, 0.21960);

    float3 color = kajiyaKay(diffuse, specular, 16.0, T, L, V);
    return float4(color, 1.0);
}
