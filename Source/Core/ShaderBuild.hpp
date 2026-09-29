#pragma once

#include <expected>
#include <filesystem>
#include <vector>
#include <slang/slang.h>
#include <slang/slang-com-ptr.h>

class ShaderBuild
{
public:
    ShaderBuild();

private:
    [[nodiscard]] static std::vector<std::filesystem::path> collectShaders();

    void createSlangSession();

    void compileShader(const std::filesystem::path& source);

    [[nodiscard]] static std::expected<std::vector<char>, std::string> readShaderFile(const std::filesystem::path& filePath);

    [[nodiscard]] static std::expected<std::string, std::string> stageToExtension(SlangStage stage);

    void writeShader(const std::filesystem::path& binPath, slang::IBlob* spirv);

    void diag(slang::IBlob* diagBlob);

    [[nodiscard]] static std::filesystem::path fullStem(std::filesystem::path p);

    uint32_t                             mSuccess = 0;
    uint32_t                             mStages = 0;
    Slang::ComPtr<slang::IGlobalSession> mGlobalSession;
    Slang::ComPtr<slang::ISession>       mSession;
};
